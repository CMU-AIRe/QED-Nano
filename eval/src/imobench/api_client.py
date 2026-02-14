"""This module provides a unified API for querying various large language models."""

import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import anthropic
import requests
from anthropic.types import TextBlock, ThinkingBlock
from loguru import logger
from openai import OpenAI, RateLimitError
from together import Together
from tqdm import tqdm


from imobench.request_logger import request_logger


class APIClient:
    """A client that queries various LLM APIs."""

    def __init__(
        self,
        model,
        timeout=18000,
        max_tokens=None,
        api="openai",
        api_key_env=None,
        base_url=None,
        max_retries=25,
        max_retries_inner=25,
        concurrent_requests=30,
        no_system_messages=False,
        read_cost=1,
        write_cost=1,
        sleep_on_error=60,
        sleep_after_request=0.1,
        background=False,
        include_max_tool_calls=True,
        throw_error_on_failure=False,
        max_tokens_param="max_tokens",
        reasoning_effort=None,
        use_openai_responses_api=False,
        delimiters=None,
        max_tool_calls=0,
        tools=None,
        **kwargs,
    ):
        """Initializes the APIClient object and params. All prompts are set at run_queries invocation.

        Args:
            model (str): The name of the model to use.
            timeout (int, optional): The timeout for API requests in seconds. Defaults to 9000.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to None.
            api (str, optional): The API to use. Defaults to 'openai'.
            max_retries (int, optional): The maximum number of retries for a failed query. Defaults to 50.
            concurrent_requests (int, optional): The number of concurrent requests to make. Defaults to 30.
            no_system_messages (bool, optional): Whether to disable system messages. Defaults to False.
            read_cost (int, optional): The cost of reading a token. Defaults to 1.
            write_cost (int, optional): The cost of writing a token. Defaults to 1.
            sleep_on_error (int, optional): The number of seconds to sleep on an error. Defaults to 60.
            sleep_after_request (float, optional): The number of seconds to sleep after a request. Defaults to 0.1.
            throw_error_on_failure (bool, optional): Whether to throw an error on failure. Defaults to False.
            max_tokens_param (str, optional): The name of the max_tokens parameter for the API. Defaults to "max_tokens".
            reasoning_effort (str, optional): The reasoning effort to use. Defaults to None.
            use_openai_responses_api (bool, optional): Whether to use OpenAI responses. Defaults to False.
            max_tool_calls (int|dict, optional): The maximum number of tool calls to make. Defaults to 0.
                Could also be a dict that specifies max calls per tool name.
            tools (list, optional): A list of tools to use. Defaults to None.
            **kwargs: Additional keyword arguments for the API.
        """
        # Max tool calls
        self.tool_calls_allowed = False
        if isinstance(max_tool_calls, int):
            self.max_tool_calls = {"any": 0}
            self.max_tool_calls_mode = "total"
            if max_tool_calls > 0:
                self.tool_calls_allowed = True
        elif isinstance(max_tool_calls, dict):
            self.max_tool_calls = max_tool_calls
            self.max_tool_calls_mode = "per_tool"
            if sum(max_tool_calls.values()) > 0:
                self.tool_calls_allowed = True

        # Adapt model name and other args to the model
        if "--" in model:
            model, reasoning_effort = model.split("--")
            logger.info(f"Model: {model}, Reasoning effort: {reasoning_effort}")
        if ("o1" in model or "o3" in model or "o4" in model or "gpt-5" in model) and api == "openai":
            no_system_messages = True  # o1 model cannot handle system messages
            if not use_openai_responses_api:
                max_tokens_param = "max_completion_tokens"
        if use_openai_responses_api:
            max_tokens_param = "max_output_tokens"
        if self.tool_calls_allowed and not use_openai_responses_api:
            max_tokens_param = "max_completion_tokens"
        self._kwarg_remover(api, model, kwargs)

        self.model = model
        self.kwargs = kwargs
        if max_tokens is not None:
            self.kwargs[max_tokens_param] = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_retries_inner = max_retries_inner
        self.throw_error_on_failure = throw_error_on_failure
        self.concurrent_requests = concurrent_requests
        self.no_system_messages = no_system_messages
        self.sleep_on_error = sleep_on_error
        self.sleep_after_request = sleep_after_request
        self.read_cost = read_cost
        self.write_cost = write_cost
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.use_openai_responses_api = use_openai_responses_api
        self.include_max_tool_calls = include_max_tool_calls
        self.delimiters = delimiters
        if isinstance(self.delimiters, str):
            self.delimiters = [self.delimiters]
        self.background = background
        if max_tokens is not None:
            self.max_tokens_param = max_tokens_param
        if reasoning_effort is not None:
            if not self.use_openai_responses_api:
                self.kwargs["reasoning_effort"] = reasoning_effort
            elif "reasoning" in self.kwargs:
                self.kwargs["reasoning"]["effort"] = reasoning_effort
            else:
                self.kwargs["reasoning"] = {"effort": reasoning_effort}

        # Save tools: user should forward all (even if mix of competition-given and scaffold-given)
        self.tools = tools if tools is not None else []
        self.tool_functions = {
            tool_desc["function"]["name"]: func for func, tool_desc in self.tools if "function" in tool_desc
        }
        self.tool_descriptions = [tool_desc for _, tool_desc in self.tools]
        if (not self.tool_calls_allowed or len(self.tool_descriptions) == 0) and "tool_choice" in self.kwargs:
            del self.kwargs["tool_choice"]

        # Prep api
        self.api = api
        self.api_key = None
        self.terminated = False
        self._initialize_api_keys()

    def terminate(self):
        """Terminates the APIClient."""
        self.terminated = True

    def _kwarg_remover(self, api, model, kwargs):
        """Removes kwargs that are not supported by the API or model.

        Args:
            api (str): The API to use.
            model (str): The model to use.
            kwargs (dict): The kwargs to clean.
        """
        if "human_readable_id" in kwargs:
            del kwargs["human_readable_id"]
        if "date" in kwargs:
            del kwargs["date"]
        if "prompt_margin" in kwargs:
            del kwargs["prompt_margin"]
        if "model_revision" in kwargs:
            del kwargs["model_revision"]
        if "use_openai_responses_api_tools" in kwargs:
            del kwargs["use_openai_responses_api_tools"]
        if any([kw in model for kw in ["o1", "o3", "o4"]]) and "temperature" in kwargs:
            del kwargs["temperature"]
        for kwarg in ["top_p", "top_k", "temperature"]:
            if kwarg in kwargs and kwargs[kwarg] is None:
                del kwargs[kwarg]
        if (api == "anthropic" and "claude-3-7" in model) or (("o1" in model or "o3" in model) and api == "openai"):
            for kwarg_to_remove in ["top_p", "top_k", "temperature"]:
                if kwarg_to_remove in kwargs:
                    logger.info(f"Removing {kwarg_to_remove} parameter for {model} model.")
                    del kwargs[kwarg_to_remove]

    def _initialize_api_keys(self):
        """Initializes the API keys and base URLs for the selected API."""
        if self.api == "xai":
            self.api_key = os.getenv("XAI_API_KEY")
            self.base_url = "https://api.x.ai/v1"
            self.api = "openai"
        elif self.api == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
        elif self.api == "together":
            self.api_key = os.getenv("TOGETHER_API_KEY")
            self.base_url = "https://api.together.xyz/v1"
        elif self.api == "google":
            self.api_key = os.getenv("GOOGLE_API_KEY")
            self.api = "openai"  # !
            self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        elif self.api == "anthropic":
            if self.tool_calls_allowed:
                self.api = "openai"
                self.base_url = "https://api.anthropic.com/v1/"
                if "thinking" in self.kwargs:
                    self.kwargs["extra_body"] = self.kwargs["thinking"]
                    # TODO discuss bug
                    del self.kwargs["thinking"]
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        elif self.api == "glm":
            self.api_key = os.getenv("GLM_API_KEY")
            self.base_url = "https://api.z.ai/api/paas/v4/"
            self.api = "openai"
        elif self.api == "deepseek":
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
            self.base_url = "https://api.deepseek.com"
            self.api = "openai"
        elif self.api == "openrouter":
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            self.base_url = "https://openrouter.ai/api/v1"
            self.api = "openai"
        elif self.api == "custom":
            self.api = "openai"
            self.base_url = self.base_url
            self.api_key = os.getenv(self.api_key_env) if self.api_key_env is not None else None
        else:
            raise ValueError(f"API {self.api} not supported.")

        assert self.api_key is not None or self.api == "custom", "API key not found."

    class InternalRequestResult:
        """A class to hold the result of a request internally (below run_queries)."""

        def __init__(self, conversation, input_tokens, output_tokens):
            self.conversation = conversation
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

    def run_queries(self, queries, no_tqdm=False, ignore_tool_calls=False, custom_indices=None):
        """Only entry point: runs a given list of queries through the API.

        Args:
            queries (list[MessageList]): A list of queries to run. Each query is a MessageList that we will mangle into
            the right format for this API.
            no_tqdm (bool, optional): Whether to disable the tqdm progress bar. Defaults to False.
            ignore_tool_calls (bool, optional): Whether to ignore tool calls in this interaction. Defaults to False.

        Yields:
            tuple: An (idx, conversation, detailed_cost) tuple.
                idx: Integer index of the query this response corresponds to in [0, len(queries)-1].
                conversation: Full list of messages (including those from the query) in the API format (incl. CoT).
                detailed_cost: A dict with total "cost" ($), "input_tokens", "output_tokens", and "time" (seconds).
        """
        if not no_tqdm:
            logger.info(f"Running {len(queries)} queries.")

        # For now only switches between system/developer, keeps rest intact
        queries = [self._validate_and_prepare_query(query) for query in queries]

        # Prepare batch indices, agents use custom ones for request_logger
        if custom_indices is not None:
            if len(custom_indices) != len(queries):
                logger.warning(
                    "custom_indices length ({}) does not match queries length ({}); falling back to default indices.",
                    len(custom_indices),
                    len(queries),
                )
                indices = list(range(len(queries)))
            else:
                indices = list(custom_indices)
        else:
            indices = list(range(len(queries)))

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.concurrent_requests) as executor:
            future_to_index = {}
            future_to_query_index = {}
            for query_index, (idx, query) in enumerate(zip(indices, queries)):
                future = executor.submit(self._run_query_with_retry, idx, query, ignore_tool_calls)
                future_to_index[future] = idx
                future_to_query_index[future] = query_index

            iterator = as_completed(future_to_index)
            if not no_tqdm:
                iterator = tqdm(iterator, total=len(future_to_index))

            for future in iterator:
                idx = future_to_index[future]
                query_index = future_to_query_index[future]
                result = future.result()
                if result is None:
                    if 0 <= query_index < len(queries):
                        base_query = queries[query_index]
                    else:
                        logger.warning(
                            "Query index {} out of range for {} queries; returning empty assistant response.",
                            query_index,
                            len(queries),
                        )
                        base_query = []
                    conversation = [m.copy() for m in base_query] + [{"role": "assistant", "content": ""}]
                    result = self.InternalRequestResult(conversation, input_tokens=0, output_tokens=0)
                detailed_cost = {
                    "cost": self._get_cost(result.input_tokens, result.output_tokens),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "time": time.time() - start_time,
                }
                yield idx, result.conversation, detailed_cost

    def _validate_and_prepare_query(self, query):
        """Prepares a query for the API.
            All "tool_response" and "assistant" blocks must have come straight from this APIClient
                => We only need to normalize the developer and user messages
            We will assume they arrive in normalized format (only role: "user"/"developer" and content:str fields)

        Args:
            query (MessageList): List of messages to prepare.

        Returns:
            query_prepared (MessageList): The prepared conversation in the format for this API.
        """
        query_prepared = []
        for m in query:
            query_prepared.append(m.copy())
            if m.get("role", "") == "developer" and not self.no_system_messages:
                query_prepared[-1]["role"] = "system"  # use system if expected by API
        return query_prepared

    def _get_cost(self, input_tokens, output_tokens):
        return (input_tokens * self.read_cost + output_tokens * self.write_cost) / 1e6

    def _get_messages_from_anthropic_content(self, content):
        """Postprocesses the content from an Anthropic API query.

        Args:
            content: The content from the Anthropic API.

        Returns:
            str: The textual representation.
        """
        messages = []
        for content_block in content:
            if isinstance(content_block, ThinkingBlock):
                messages.append({"role": "assistant", "type": "reasoning", "content": content_block.thinking})
            elif isinstance(content_block, TextBlock):
                messages.append({"role": "assistant", "type": "response", "content": content_block.text})
                break
        return messages

    def _drop_cot(self, messages):
        """Drops all CoT/thinking/reasoning messages from a conversation.
        This is a cost saving measure at API call site; conversations that are maintained will have this.

        Args:
            messages (MessageList): The conversation to drop CoT from.
        Returns:
            MessageList: The conversation without CoT messages.
        """
        new_messages = []
        for m in messages:
            if m.get("role", "") == "assistant" and m.get("type", "response") == "cot":
                continue
            new_messages.append(m)

        return new_messages
    """
        Case 2: Standard API
    """
    def _run_query_with_retry(self, idx, query, ignore_tool_calls=False):
        """Runs a query on standard API with retries on failure.

        Args:
            idx (int): The index of the query in the batch of queries given to run_queries.
            query (MessageList): The query to run.
            ignore_tool_calls (bool, optional): Whether to ignore tool calls in this interaction.

        Returns:
            InternalRequestResult or None
        """
        retry_idx = 0
        while retry_idx < self.max_retries:
            if self.terminated:
                return None
            try:
                result = self._run_query(idx, query, ignore_tool_calls=ignore_tool_calls)
                time.sleep(self.sleep_after_request)
                return result
            except Exception as e:
                logger.error(f"Error in outer retries. Exception: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                time.sleep(self.sleep_on_error)
                # if api error is not due to rate limit, try again
                if "rate limit" not in str(e).lower() and "429" not in str(e):
                    retry_idx += 1
                if "violating our usage policy" in str(e).lower():
                    print("Stopping - prompt repeatedly violated usage policy -- ", query)
                    if retry_idx > 3:
                        break
                continue
        if self.throw_error_on_failure:
            raise ValueError("Max outer retries reached.")
        else:
            return None

    def _run_query(self, idx, query, ignore_tool_calls=False):
        """Runs a query on standard API.

        Args:
            idx (int): The index of the query in the batch of queries given to run_queries.
            query (MessageList): The query to run.
            ignore_tool_calls (bool, optional): Whether to ignore tool calls in this interaction.

        Returns:
            InternalRequestResult or None
        """
        if self.api == "openai":
            return self._openai_query_with_tools(idx, query, ignore_tool_calls=ignore_tool_calls)
        elif self.api == "together":
            return self._openai_query_with_tools(idx, query, is_together=True, ignore_tool_calls=ignore_tool_calls)
        elif self.api == "anthropic":
            return self._anthropic_query(idx, query)

    def _anthropic_query(self, idx, query):
        """Queries the Anthropic API.

        Args:
            idx (int): The index of the query in the batch of queries given to run_queries.
            query (MessageList): The query to run.

        Returns:
            InternalRequestResult or None
        """
        client = anthropic.Anthropic(
            api_key=self.api_key,
            max_retries=0,
            timeout=self.timeout,
        )
        system_message = anthropic.NOT_GIVEN
        if query[0]["role"] == "system":
            system_message = query[0]["content"]
            query = query[1:]
        raw_result = client.messages.create(
            model=self.model, messages=self._drop_cot(query), system=system_message, **self.kwargs
        )

        new_messages = self._get_messages_from_anthropic_content(raw_result.content)
        conversation = [m.copy() for m in query] + new_messages
        input_tokens = raw_result.usage.input_tokens
        output_tokens = raw_result.usage.output_tokens
        return self.InternalRequestResult(conversation, input_tokens, output_tokens)

    def _openai_query_with_tools(self, idx, query, is_together=False, ignore_tool_calls=False):
        """Queries the OpenAI API with tools.

        Args:
            idx (int): The index of the query in the batch of queries given to run_queries.
            query (MessageList): The query to run.
            is_together (bool, optional): Whether to use the Together API. Defaults to False.
            ignore_tool_calls (bool, optional): Whether to ignore tool calls in this interaction.

        Returns:
            InternalRequestResult or None
        """
        if is_together:
            client = Together()
        else:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout, max_retries=0)

        if self.use_openai_responses_api:
            return self._openai_query_responses_api(client, idx, query, ignore_tool_calls=ignore_tool_calls)
        else:
            return self._openai_query_chat_completions_api(client, idx, query, ignore_tool_calls=ignore_tool_calls)

    def _openai_query_responses_api(self, client, idx, messages, ignore_tool_calls=False):
        """Queries the OpenAI API with the responses API.

        Args:
            client: The OpenAI client.
            idx (int): The index of the query in the batch of queries given to run_queries.
            messages (list): The messages to send.
            ignore_tool_calls (bool, optional): Whether to ignore tool calls in this interaction.

        Returns:
            InternalRequestResult or None
        """

        # Set up tools
        response_tools = []
        for tool_desc in self.tool_descriptions:
            if tool_desc["type"] != "function":
                response_tools.append(tool_desc)
            else:
                response_tools.append({"type": "function", **tool_desc["function"]})
        if ignore_tool_calls:
            max_tool_calls_mode, max_tool_calls = "total", {"any": 0}
        elif len(response_tools) == 1 and response_tools[0]["type"] == "code_interpreter":
            max_tool_calls_mode, max_tool_calls = "total", {"any": 0}
        else:
            max_tool_calls_mode, max_tool_calls = self.max_tool_calls_mode, self.max_tool_calls

        # State
        total_max_tool_calls = sum(max_tool_calls.values())
        if max_tool_calls_mode == "total":
            nb_executed_tool_calls = {"any": 0}
        else:
            nb_executed_tool_calls = {t: 0 for t in self.tool_functions.keys()}
        conversation = [m.copy() for m in messages]
        input_tokens = 0
        output_tokens = 0

        for _ in range(total_max_tool_calls + 1):
            # Inner retry to get a response
            response = None
            n_retries = 0
            while response is None and n_retries < self.max_retries_inner:
                n_retries += 1
                try:
                    payload = {
                        "model": self.model,
                        "tools": response_tools,
                        "input": self._drop_cot(conversation),  # Drop CoT here to save cost (stays in convo)
                        "timeout": self.timeout,
                        **self.kwargs,
                    }
                    if self.background:
                        payload["background"] = self.background
                    ts = time.strftime("%m%d-%H:%M:%S", time.localtime(time.time()))
                    milliseconds = int((time.time() % 1) * 1000)
                    ts += f".{datetime.now().microsecond:06d}"
                    info = {"nb_executed_tool_calls": nb_executed_tool_calls, "n_retries": n_retries}
                    request_logger.log_request(ts=ts, batch_idx=idx, request=payload, **info)
                    response = client.responses.create(**payload)
                    if self.background:
                        while response.status in {"queued", "in_progress"}:
                            time.sleep(5)
                            response = client.responses.retrieve(response.id)
                    request_logger.log_response(ts=ts, batch_idx=idx, response=response.model_dump())
                except Exception as e:
                    request_logger.log_response(ts=ts, batch_idx=idx, response={"exception": str(e)})
                    time.sleep(20)
                    logger.error(f"Got OpenAI error in responses api inner. Exception: {e}")
                    continue
            if response is None:
                raise ValueError("Max inner retries reached.")

            # Update state: token counts and conversation (potentially execute tool calls)
            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens

            was_tool_call_executed = False
            for out in response.output:
                if out.type == "message":
                    for c in out.content:
                        if c.type == "output_text":
                            conversation.append({"role": "assistant", "content": c.text})
                elif out.type == "code_interpreter_call":
                    conversation.append(
                        {
                            "type": "code_interpreter_call",
                            "code": out.code,
                            "id": out.id,
                            "container_id": out.container_id,
                        }
                    )
                elif out.type == "function_call":
                    function_name = out.name
                    arguments = json.loads(out.arguments)
                    tool_func = self.tool_functions[function_name]
                    tool_key = "any" if max_tool_calls_mode == "total" else function_name
                    if nb_executed_tool_calls[tool_key] >= max_tool_calls[tool_key]:
                        output = f"Error: Tool call after exceeding max # of tool calls ({max_tool_calls[tool_key]})."
                    else:
                        try:
                            output = tool_func(**arguments)  # EXECUTE
                        except Exception as e:
                            output = f"Error executing tool {function_name}. Exception: {e}"
                    if not isinstance(output, str):
                        additional_cost = output[1]
                        input_tokens += additional_cost["input_tokens"]
                        output_tokens += additional_cost["output_tokens"]
                        output = output[0]
                    conversation.append(
                        {
                            "type": "function_call",
                            "call_id": out.call_id,
                            "arguments": out.arguments,
                            "name": out.name,
                        }
                    )
                    was_tool_call_executed = True
                    nb_executed_tool_calls[tool_key] += 1
                    nb_tool_calls_left = max_tool_calls[tool_key] - nb_executed_tool_calls[tool_key]
                    detail = "for this tool" if max_tool_calls_mode == "per_tool" else "(across all tools)"
                    if self.include_max_tool_calls:
                        info = f"\n\n### INFO ###\nYou have {nb_tool_calls_left} tool executions left {detail}."
                    else:
                        info = ""
                    conversation.append(
                        {"type": "function_call_output", "call_id": out.call_id, "output": output + info}
                    )
                elif out.type == "reasoning":
                    summary = ""
                    for thought in out.summary:
                        if thought.text is not None:
                            summary += "<thought>" + "\n" + thought.text + "\n" + "</thought>\n"
                    conversation.append({"role": "assistant", "type": "cot", "content": summary, "id": out.id})
                else:
                    raise ValueError(f"Unknown output type {out.type}")

            # If nothing was run this was the last iteration, stop
            if not was_tool_call_executed or self.terminated:
                break

        if len(conversation) == len(messages):
            conversation.append({"role": "assistant", "content": ""})
        return self.InternalRequestResult(conversation, input_tokens, output_tokens)

    def _openai_query_chat_completions_api(self, client, idx, messages, ignore_tool_calls=False):
        """Queries the OpenAI API using chat completions API.

        Args:
            client: The OpenAI client.
            idx (int): The index of the query in the batch of queries given to run_queries.
            messages (list): The messages to send.
            ignore_tool_calls (bool): Whether to ignore tool calls.

        Returns:
            InternalRequestResult or None
        """

        # Set up tools
        if ignore_tool_calls:
            max_tool_calls_mode, max_tool_calls = "total", {"any": 0}
        else:
            max_tool_calls_mode, max_tool_calls = self.max_tool_calls_mode, self.max_tool_calls

        # State
        total_max_tool_calls = sum(max_tool_calls.values())
        if max_tool_calls_mode == "total":
            nb_executed_tool_calls = {"any": 0}
        else:
            nb_executed_tool_calls = {t: 0 for t in self.tool_functions.keys()}
        conversation = [m.copy() for m in messages]
        input_tokens = 0
        output_tokens = 0
        max_output_tokens = self.kwargs.get(self.max_tokens_param, None)

        # As long as we just had a tool response do another request
        was_tool_call_executed = True
        while was_tool_call_executed and not self.terminated:
            was_tool_call_executed = False
            # Inner retry to get a response
            response = None
            n_retries = -1
            while response is None and n_retries < self.max_retries_inner:
                n_retries += 1
                try:
                    # Extract vLLM-specific params that need to go in extra_body
                    kwargs = self.kwargs.copy()
                    extra_body = {}
                    if "top_k" in kwargs:
                        extra_body["top_k"] = kwargs.pop("top_k")
                    kwargs[self.max_tokens_param] = max_output_tokens
                    payload = {
                        "model": self.model,
                        "messages": self._drop_cot(conversation),  # Drop CoT here to save cost (stays in convo)
                        "tools": self.tool_descriptions if len(self.tool_descriptions) > 0 else None,
                        "timeout": self.timeout,
                        **kwargs,
                    }
                    if extra_body:
                        payload["extra_body"] = extra_body
                    ts = time.strftime("%m%d-%H:%M:%S", time.localtime(time.time()))
                    ts += f".{datetime.now().microsecond:06d}"
                    info = {"nb_executed_tool_calls": nb_executed_tool_calls, "n_retries": n_retries}
                    request_logger.log_request(ts=ts, batch_idx=idx, request=payload, **info)
                    response = client.chat.completions.create(**payload)
                    request_logger.log_response(ts=ts, batch_idx=idx, response=response.model_dump())
                except Exception as e:
                    request_logger.log_response(ts=ts, batch_idx=idx, response={"exception": str(e)})
                    if isinstance(e, RateLimitError):
                        logger.info(f"Got OpenAI CC rate limit error. Sleeping for 60 seconds. Exception: {e}")
                        time.sleep(60)
                        continue
                    else:
                        if "maximum context length" in str(e).lower() or "input token count" in str(e).lower() or "max_tokens" in str(e).lower():
                            max_output_tokens = int(max_output_tokens / 1.2)
                            logger.info(
                                f"Got OpenAI CC max context length error. Reducing max output tokens to {max_output_tokens} and retrying. Exception: {e}"
                            )
                        logger.info(f"Got OpenAI CC non ratelimit error. Sleeping for 20 seconds: {e}")
                        time.sleep(20)
                        continue
            if response is None:
                raise ValueError("Max inner retries reached.")

            # Update state: token counts and conversation (potentially execute tool calls)
            input_tokens += response.usage.prompt_tokens
            output_tokens += response.usage.total_tokens - response.usage.prompt_tokens
            message = response.choices[0].message

            # Separate CoT from final answer
            cot_text = None
            if hasattr(message, "reasoning") and message.reasoning:
                cot_text = message.reasoning
            elif hasattr(message, "reasoning_content") and message.reasoning_content:
                cot_text = message.reasoning_content

            extracted_cot_from_content = None
            if self.delimiters is not None and hasattr(message, "content") and message.content:
                delim_found = False
                for delim in self.delimiters:
                    if delim in message.content:
                        reasoning_split = message.content.split(delim)
                        reasoning = delim.join(reasoning_split[:-1]).strip()
                        message.content = reasoning_split[-1].strip()
                        extracted_cot_from_content = reasoning
                        delim_found = True
                        break
                if not delim_found and cot_text is None:
                    # Delimiters configured but none found - treat entire content as CoT 
                    extracted_cot_from_content = message.content
                    message.content = ""
                    
            # Add CoT and rest of message separately
            if cot_text:
                conversation.append({"role": "assistant", "type": "cot", "content": cot_text})
            elif extracted_cot_from_content:
                conversation.append({"role": "assistant", "type": "cot", "content": extracted_cot_from_content})
            message_dict = message.model_dump()
            message_dict = {k: v for k, v in message_dict.items() if v is not None}  # Drop nulls
            if "reasoning" in message_dict:
                del message_dict["reasoning"]
            if "reasoning_content" in message_dict:
                del message_dict["reasoning_content"]
            conversation.append(message_dict)  # Should have tool calls inside too!

            # Try to execute all tool calls
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    if function_name not in self.tool_functions:
                        logger.warning(f"Tool {function_name} not found, skipping.")
                        continue
                    tool_func = self.tool_functions[function_name]

                    # If no budget return error
                    # NOTE: just erroring out here might stop the request loop but the model will be given last chance.
                    tool_key = "any" if max_tool_calls_mode == "total" else function_name
                    if nb_executed_tool_calls[tool_key] >= max_tool_calls[tool_key]:
                        error = f"Error: Exceeded maximum number of tool calls ({max_tool_calls[tool_key]})."
                        conversation.append({"role": "tool", "tool_call_id": tool_call.id, "content": error})
                    else:
                        # Execute tool
                        arguments = json.loads(tool_call.function.arguments)
                        try:
                            output = tool_func(**arguments)
                        except Exception as e:
                            output = f"Error executing tool {function_name}. Exception: {e}"

                        # Tools can return additional cost
                        if isinstance(output, tuple):
                            output, extra_cost = output
                            input_tokens += extra_cost["input_tokens"]
                            output_tokens += extra_cost["output_tokens"]
                        
                        nb_executed_tool_calls[tool_key] += 1

                        nb_tool_calls_left = max_tool_calls[tool_key] - nb_executed_tool_calls[tool_key]
                        detail = "for this tool" if max_tool_calls_mode == "per_tool" else "(across all tools)"
                        info = f"\n\n### INFO ###\nYou have {nb_tool_calls_left} tool executions left {detail}."
                        conversation.append(
                            {
                                "role": "tool",
                                "tool_name": function_name,
                                "tool_call_id": tool_call.id,
                                "content": output + info,
                            }
                        )

                    was_tool_call_executed = True
                    
                    
        if total_max_tool_calls > 0:
            logger.info(f"Finished on a loop without tool calls, after executing {nb_executed_tool_calls} calls total.")

        return self.InternalRequestResult(conversation, input_tokens, output_tokens)
