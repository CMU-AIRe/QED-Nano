let isSidebarCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';

document.addEventListener('DOMContentLoaded', () => {
    const sidebar = document.getElementById('sidebar');
    const toggleButton = document.getElementById('sidebar-toggle-button');
    const filterForm = document.getElementById('filter-form');

    if (isSidebarCollapsed && sidebar) {
        sidebar.classList.add('collapsed');
        if (toggleButton) toggleButton.innerHTML = '&#9776;'; // Hamburger icon
    } else if (sidebar) {
        sidebar.classList.remove('collapsed');
        if (toggleButton) toggleButton.innerHTML = '&times;'; // Close icon
    }

    setupDatasetForms();

    if (filterForm) {
        const datasetName = filterForm.dataset.datasetName;
        // Initial load
        filterForm.dispatchEvent(new Event('submit'));
        // Load the first problem by default
        loadProblem(0, datasetName);
    }
});

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const toggleButton = document.getElementById('sidebar-toggle-button');

    isSidebarCollapsed = !isSidebarCollapsed;
    localStorage.setItem('sidebarCollapsed', isSidebarCollapsed); // Persist state

    if (isSidebarCollapsed) {
        sidebar.classList.add('collapsed');
        if (toggleButton) toggleButton.innerHTML = '&#9776;'; // Hamburger icon
    } else {
        sidebar.classList.remove('collapsed');
        if (toggleButton) toggleButton.innerHTML = '&times;'; // Close icon
    }
}

const filterForm = document.getElementById('filter-form');
if (filterForm) {
    filterForm.addEventListener('submit', function(e) {
        e.preventDefault();

        const formData = new FormData(this);
        const params = new URLSearchParams();
        const datasetName = this.dataset.datasetName;

        for (let [name, value] of formData.entries()) {
            if (value) {
                params.append(name, value);
            }
        }

        fetch(`/data/${datasetName}?${params.toString()}`)
            .then(response => response.json())
            .then(data => {
                const resultsContainer = document.getElementById('sidebar-results');
                resultsContainer.innerHTML = ''; // Clear previous results

                if (data.sidebar_metadata) {
                    data.sidebar_metadata.forEach(meta => {
                        const item = document.createElement('div');
                        item.classList.add('sidebar-item');
                        item.setAttribute('data-id', meta.idx);

                        item.innerHTML = meta.display_string;

                        item.addEventListener('click', function() {
                            loadProblem(meta.idx, datasetName);
                        });

                        resultsContainer.appendChild(item);
                    });
                }
            });
    });
}

function setupDatasetForms() {
    const forms = document.querySelectorAll('[data-dataset-form]');
    if (!forms.length) return;

    const datasetCache = new Map();
    const datasetInfoCache = new Map();

    forms.forEach(form => {
        const datasetInput = form.querySelector('[data-dataset-input]');
        const configInput = form.querySelector('[data-config-input]');
        const splitInput = form.querySelector('[data-split-input]');
        const suggestionBox = form.querySelector('[data-dataset-suggestions]');

        if (!datasetInput || !configInput || !splitInput || !suggestionBox) return;

        const updateSelectOptions = (selectEl, options, placeholder) => {
            if (!selectEl) return;
            const currentValue = selectEl.value;
            selectEl.innerHTML = '';
            const placeholderOption = document.createElement('option');
            placeholderOption.value = '';
            placeholderOption.textContent = placeholder;
            selectEl.appendChild(placeholderOption);
            options.forEach(option => {
                const opt = document.createElement('option');
                opt.value = option;
                opt.textContent = option;
                selectEl.appendChild(opt);
            });
            if (currentValue && options.includes(currentValue)) {
                selectEl.value = currentValue;
            }
        };

        const updateSuggestions = (options) => {
            suggestionBox.innerHTML = '';
            if (!options.length) {
                suggestionBox.classList.remove('is-visible');
                return;
            }
            options.forEach(option => {
                const item = document.createElement('div');
                item.className = 'dataset-suggestion-item';
                item.textContent = option;
                item.addEventListener('mousedown', () => {
                    datasetInput.value = option;
                    suggestionBox.classList.remove('is-visible');
                    refreshConfigAndSplit();
                });
                suggestionBox.appendChild(item);
            });
            suggestionBox.classList.add('is-visible');
        };

        let searchTimeout = null;
        datasetInput.addEventListener('input', () => {
            const query = datasetInput.value.trim();
            if (searchTimeout) window.clearTimeout(searchTimeout);
            searchTimeout = window.setTimeout(async () => {
                if (!query) {
                    updateSuggestions([]);
                    return;
                }

                if (datasetCache.has(query)) {
                    updateSuggestions(datasetCache.get(query));
                    return;
                }

                try {
                    const response = await fetch(`/datasets?search=${encodeURIComponent(query)}`);
                    const data = await response.json();
                    const datasets = data.datasets || [];
                    datasetCache.set(query, datasets);
                    updateSuggestions(datasets);
                } catch (err) {
                    updateSuggestions([]);
                }
            }, 250);
        });

        const fetchDatasetInfo = async (configName = '') => {
            const datasetName = datasetInput.value.trim();
            if (!datasetName) return null;
            const cacheKey = `${datasetName}::${configName}`;
            if (datasetInfoCache.has(cacheKey)) {
                return datasetInfoCache.get(cacheKey);
            }

            try {
                const params = new URLSearchParams({ name: datasetName });
                if (configName) params.set('config', configName);
                const response = await fetch(`/dataset-info?${params.toString()}`);
                const data = await response.json();
                datasetInfoCache.set(cacheKey, data);
                return data;
            } catch (err) {
                return null;
            }
        };

        const refreshConfigAndSplit = async () => {
            const info = await fetchDatasetInfo();
            if (!info) return;
            updateSelectOptions(configInput, info.configs || [], 'Select config (optional)');

            if (!configInput.value && info.configs && info.configs.length === 1) {
                configInput.value = info.configs[0];
            }

            const updated = await fetchDatasetInfo(configInput.value.trim());
            if (updated) {
                updateSelectOptions(splitInput, updated.splits || [], 'Select split (default: train)');
            }
        };

        datasetInput.addEventListener('change', refreshConfigAndSplit);
        configInput.addEventListener('change', async () => {
            const info = await fetchDatasetInfo(configInput.value.trim());
            if (info) {
                updateSelectOptions(splitInput, info.splits || [], 'Select split (default: train)');
            }
        });

        datasetInput.addEventListener('focus', () => {
            if (suggestionBox.children.length) {
                suggestionBox.classList.add('is-visible');
            }
        });

        datasetInput.addEventListener('blur', () => {
            window.setTimeout(() => suggestionBox.classList.remove('is-visible'), 150);
        });
    });
}

function loadProblem(id, datasetName) {
    fetch(`/problem/${datasetName}/${id}`)
        .then(response => response.text())
        .then(html => {
            document.getElementById('problem-container').innerHTML = html;
            renderMathInElement(document.body, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false },
                    { left: '\\(', right: '\\)', display: false },
                    { left: '\\[', right: '\\]', display: true },
                ]
            });
            hljs.highlightAll();
            // scroll to top
            window.scrollTo(0, 0);
        });
}
