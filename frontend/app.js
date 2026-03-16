const STORAGE_KEY = "oncovision.history";
const routeViews = {
    landing: document.getElementById("landingPage"),
    upload: document.getElementById("uploadPage"),
    analysis: document.getElementById("analysisPage"),
    results: document.getElementById("resultsPage"),
    history: document.getElementById("historyPage"),
    technical: document.getElementById("technicalPage"),
};

const state = {
    history: loadHistory(),
    selectedFiles: [],
    currentScan: null,
    analysisTimer: null,
    analysisStepTimer: null,
    analysisResultById: new Map(),
    activeModal: null,
};

const uploadInput = document.getElementById("ctScan");
const uploadZone = document.getElementById("uploadZone");
const uploadForm = document.getElementById("uploadForm");
const selectedFilePanel = document.getElementById("selectedFilePanel");
const selectedFileName = document.getElementById("selectedFileName");
const selectedFileMeta = document.getElementById("selectedFileMeta");
const fileFormatValue = document.getElementById("fileFormatValue");
const fileCountValue = document.getElementById("fileCountValue");
const fileSizeValue = document.getElementById("fileSizeValue");
const removeFileButton = document.getElementById("removeFileButton");
const analyzeButton = document.getElementById("analyzeButton");
const sampleButton = document.getElementById("sampleButton");
const uploadErrorBanner = document.getElementById("uploadErrorBanner");
const recentUploads = document.getElementById("recentUploads");
const analysisDetail = document.getElementById("analysisDetail");
const analysisProgressBar = document.getElementById("analysisProgressBar");
const analysisProgressLabel = document.getElementById("analysisProgressLabel");
const analysisSteps = Array.from(document.querySelectorAll(".analysis-steps span"));
const riskBanner = document.getElementById("riskBanner");
const riskIcon = document.getElementById("riskIcon");
const riskLabel = document.getElementById("riskLabel");
const riskSummary = document.getElementById("riskSummary");
const riskRecommendation = document.getElementById("riskRecommendation");
const noduleCount = document.getElementById("noduleCount");
const analysisDate = document.getElementById("analysisDate");
const analysisStatus = document.getElementById("analysisStatus");
const resultsMeta = document.getElementById("resultsMeta");
const scanFileName = document.getElementById("scanFileName");
const scanFileSize = document.getElementById("scanFileSize");
const scanFileFormat = document.getElementById("scanFileFormat");
const recommendationText = document.getElementById("recommendationText");
const timingGrid = document.getElementById("timingGrid");
const scanImage = document.getElementById("scanImage");
const scanPlaceholder = document.getElementById("scanPlaceholder");
const nodulesList = document.getElementById("nodulesList");
const emptyState = document.getElementById("emptyState");
const noduleFilter = document.getElementById("noduleFilter");
const resultsBreadcrumb = document.getElementById("resultsBreadcrumb");
const historySubtitle = document.getElementById("historySubtitle");
const historySearch = document.getElementById("historySearch");
const historyFilter = document.getElementById("historyFilter");
const historySort = document.getElementById("historySort");
const historyGroups = document.getElementById("historyGroups");
const noduleCardTemplate = document.getElementById("noduleCardTemplate");
const historyCardTemplate = document.getElementById("historyCardTemplate");
const exportImageButton = document.getElementById("exportImageButton");
const exportPdfButton = document.getElementById("exportPdfButton");
const floatingExportButton = document.getElementById("floatingExportButton");
const shareButton = document.getElementById("shareButton");
const saveScanButton = document.getElementById("saveScanButton");
const deleteScanButton = document.getElementById("deleteScanButton");
const previousScanButton = document.getElementById("previousScanButton");
const nextScanButton = document.getElementById("nextScanButton");
const mobileNavToggle = document.getElementById("mobileNavToggle");
const mobileNav = document.getElementById("mobileNav");
const topbar = document.getElementById("topbar");
const modalBackdrop = document.getElementById("modalBackdrop");
const settingsModal = document.getElementById("settingsModal");
const helpModal = document.getElementById("helpModal");
const exportModal = document.getElementById("exportModal");
const shareModal = document.getElementById("shareModal");
const technicalTabs = Array.from(document.querySelectorAll("[data-tech-tab]"));
const previewExportButton = document.getElementById("previewExportButton");
const patientNameInput = document.getElementById("patientNameInput");
const patientMrnInput = document.getElementById("patientMrnInput");
const patientDobInput = document.getElementById("patientDobInput");
const physicianNotesInput = document.getElementById("physicianNotesInput");
const confirmShareButton = document.getElementById("confirmShareButton");
const shareEmailInput = document.getElementById("shareEmailInput");
const shareMessageInput = document.getElementById("shareMessageInput");
const themeRadios = Array.from(document.querySelectorAll("input[name='theme']"));
const THEME_KEY = "oncovision.theme";
const prefersDarkScheme = window.matchMedia("(prefers-color-scheme: dark)");

const analysisMessages = [
    "Processing 324 slices",
    "Detecting nodules",
    "Calculating risk",
];

wireNavigation();
wireUpload();
wireActions();
wireModals();
wireTechnicalTabs();
wireThemeSelector();
renderRecentUploads();
renderHistoryPage();
updateNavState();
handleRoute();
window.addEventListener("popstate", handleRoute);
window.addEventListener("scroll", () => {
    topbar.classList.toggle("topbar--solid", window.scrollY > 24);
});

function wireNavigation() {
    document.querySelectorAll("[data-link]").forEach((link) => {
        link.addEventListener("click", (event) => {
            const href = link.getAttribute("href");
            if (!href || href.startsWith("http")) {
                return;
            }

            event.preventDefault();
            mobileNav.hidden = true;
            mobileNav.classList.remove("is-open");
            navigate(href);
        });
    });

    mobileNavToggle.addEventListener("click", () => {
        const willOpen = mobileNav.hidden;
        mobileNav.hidden = !willOpen;
        mobileNav.classList.toggle("is-open", willOpen);
    });
}

function wireUpload() {
    uploadInput.addEventListener("change", () => {
        state.selectedFiles = Array.from(uploadInput.files || []);
        syncSelectedFiles();
    });

    ["dragenter", "dragover"].forEach((eventName) => {
        uploadZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            uploadZone.classList.add("is-dragging");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        uploadZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            uploadZone.classList.remove("is-dragging");
        });
    });

    uploadZone.addEventListener("drop", (event) => {
        const files = Array.from(event.dataTransfer?.files || []);
        if (!files.length) {
            return;
        }

        state.selectedFiles = files;
        uploadInput.files = event.dataTransfer.files;
        syncSelectedFiles();
    });

    removeFileButton.addEventListener("click", () => {
        state.selectedFiles = [];
        uploadInput.value = "";
        syncSelectedFiles();
    });

    uploadForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        if (!state.selectedFiles.length) {
            showUploadError("Select at least one scan file before starting analysis.");
            return;
        }

        hideUploadError();
        const scanId = createScanId();
        const fileSummary = createFileSummary(state.selectedFiles);
        const scan = {
            id: scanId,
            fileSummary,
            createdAt: new Date().toISOString(),
            status: "Analyzing",
        };

        state.currentScan = scan;
        navigate(`/analysis/${scanId}`);
        startAnalysis(scan, async () => {
            try {
                const result = await analyzeSelectedFiles();
                finalizeAnalysis(scan, result);
            } catch (error) {
                stopAnalysisAnimation();
                navigate("/upload");
                showUploadError(error.message || "Analysis could not be completed.");
            }
        });
    });

    sampleButton.addEventListener("click", () => {
        const scanId = createScanId();
        const sample = buildSampleScan(scanId);
        state.currentScan = sample;
        navigate(`/analysis/${scanId}`);
        startAnalysis(sample, () => {
            window.setTimeout(() => {
                finalizeAnalysis(sample, sample.result);
            }, 2800);
        });
    });
}

function wireActions() {
    noduleFilter.addEventListener("change", () => {
        if (state.currentScan?.result) {
            renderResultsPage(state.currentScan);
        }
    });

    historySearch.addEventListener("input", renderHistoryPage);
    historyFilter.addEventListener("change", renderHistoryPage);
    historySort.addEventListener("change", renderHistoryPage);

    exportImageButton.addEventListener("click", exportCurrentImage);
    exportPdfButton.addEventListener("click", () => openModal("export"));
    floatingExportButton.addEventListener("click", () => openModal("export"));
    shareButton.addEventListener("click", () => openModal("share"));
    saveScanButton.addEventListener("click", () => {
        if (state.currentScan) {
            toastLikeAlert("Scan saved to local history.");
        }
    });
    deleteScanButton.addEventListener("click", deleteCurrentScan);

    previousScanButton.addEventListener("click", () => navigateHistoryNeighbor(-1));
    nextScanButton.addEventListener("click", () => navigateHistoryNeighbor(1));
}

function wireModals() {
    document.getElementById("helpButton").addEventListener("click", () => openModal("help"));
    document.getElementById("footerHelpButton").addEventListener("click", () => openModal("help"));
    document.getElementById("footerSettingsButton").addEventListener("click", () => openModal("settings"));
    previewExportButton.addEventListener("click", () => generatePdfReport(true));
    document.getElementById("generateExportButton").addEventListener("click", () => {
        generatePdfReport(false);
    });
    confirmShareButton.addEventListener("click", confirmShare);

    modalBackdrop.addEventListener("click", (event) => {
        if (event.target === modalBackdrop) {
            openModal(null);
        }
    });

    document.querySelectorAll("[data-close-modal]").forEach((button) => {
        button.addEventListener("click", () => openModal(null));
    });
}

function wireThemeSelector() {
    const stored = localStorage.getItem(THEME_KEY) || "auto";
    themeRadios.forEach((input) => {
        input.checked = input.value === stored;
        input.addEventListener("change", () => {
            localStorage.setItem(THEME_KEY, input.value);
            applyTheme(input.value);
        });
    });

    applyTheme(stored);
    const handleSystemThemeChange = () => {
        if (localStorage.getItem(THEME_KEY) === "auto") {
            applyTheme("auto");
        }
    };

    if (typeof prefersDarkScheme.addEventListener === "function") {
        prefersDarkScheme.addEventListener("change", handleSystemThemeChange);
    } else if (typeof prefersDarkScheme.addListener === "function") {
        prefersDarkScheme.addListener(handleSystemThemeChange);
    }
}

function applyTheme(preference) {
    const resolved =
        preference === "dark"
            ? "dark"
            : preference === "light"
                ? "light"
                : prefersDarkScheme.matches
                    ? "dark"
                    : "light";

    document.body.classList.toggle("theme-dark", resolved === "dark");
    document.body.classList.toggle("theme-light", resolved === "light");
    document.documentElement.dataset.theme = preference;
}

function wireTechnicalTabs() {
    technicalTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            const target = tab.dataset.techTab;
            technicalTabs.forEach((item) => item.classList.toggle("is-active", item === tab));
            document.querySelectorAll(".technical-panel").forEach((panel) => {
                panel.hidden = panel.id !== `technicalPanel-${target}`;
            });
        });
    });

    document.querySelectorAll(".code-copy").forEach((button) => {
        button.addEventListener("click", async () => {
            const target = document.getElementById(button.dataset.copyTarget);
            if (!target) {
                return;
            }

            try {
                await navigator.clipboard.writeText(target.textContent || "");
                toastLikeAlert("Code copied to clipboard.");
            } catch {
                toastLikeAlert(target.textContent || "");
            }
        });
    });
}

function navigate(path) {
    window.history.pushState({}, "", path);
    handleRoute();
}

function handleRoute() {
    const route = parseRoute(window.location.pathname);
    stopAnalysisAnimation();

    Object.values(routeViews).forEach((view) => {
        view.hidden = true;
    });

    if (route.name === "upload") {
        routeViews.upload.hidden = false;
        syncSelectedFiles();
        renderRecentUploads();
    } else if (route.name === "analysis") {
        routeViews.analysis.hidden = false;
        const scan = resolveScan(route.scanId);
        if (!scan) {
            navigate("/upload");
            return;
        }

        state.currentScan = scan;
        if (scan.result) {
            navigate(`/results/${scan.id}`);
            return;
        }
    } else if (route.name === "results") {
        routeViews.results.hidden = false;
        const scan = resolveScan(route.scanId);
        if (!scan || !scan.result) {
            navigate("/history");
            return;
        }

        state.currentScan = scan;
        renderResultsPage(scan);
    } else if (route.name === "history") {
        routeViews.history.hidden = false;
        renderHistoryPage();
    } else if (route.name === "technical") {
        routeViews.technical.hidden = false;
    } else {
        routeViews.landing.hidden = false;
    }

    updateNavState();
    window.scrollTo({ top: 0, behavior: "smooth" });
}

function parseRoute(pathname) {
    if (pathname === "/upload") {
        return { name: "upload" };
    }

    if (pathname === "/history") {
        return { name: "history" };
    }

    if (pathname === "/technical") {
        return { name: "technical" };
    }

    const analysisMatch = pathname.match(/^\/analysis\/([^/]+)$/);
    if (analysisMatch) {
        return { name: "analysis", scanId: analysisMatch[1] };
    }

    const resultsMatch = pathname.match(/^\/results\/([^/]+)$/);
    if (resultsMatch) {
        return { name: "results", scanId: resultsMatch[1] };
    }

    return { name: "landing" };
}

function updateNavState() {
    const pathname = window.location.pathname;
    document.querySelectorAll("[data-nav]").forEach((link) => {
        const key = link.dataset.nav;
        const isActive =
            (key === "home" && pathname === "/") ||
            (key === "upload" && pathname.startsWith("/upload")) ||
            (key === "history" && pathname.startsWith("/history")) ||
            (key === "technical" && pathname.startsWith("/technical"));
        link.classList.toggle("is-active", isActive);
    });
}

function syncSelectedFiles() {
    if (!state.selectedFiles.length) {
        selectedFilePanel.hidden = true;
        return;
    }

    const summary = createFileSummary(state.selectedFiles);
    selectedFilePanel.hidden = false;
    selectedFileName.textContent = formatDisplayFileName(summary.name);
    selectedFileName.title = summary.name;
    selectedFileMeta.textContent = `${summary.size} • ${summary.format}`;
    selectedFileMeta.title = `${summary.name} • ${summary.size} • ${summary.format}`;
    fileFormatValue.textContent = summary.format;
    fileCountValue.textContent = String(summary.fileCount);
    fileSizeValue.textContent = summary.size;
}

function createFileSummary(files) {
    const name = files.length === 1 ? files[0].name : `${files.length} files selected`;
    const sizeBytes = files.reduce((sum, file) => sum + file.size, 0);
    const extension = formatFileTypeLabel(files[0]?.name || "");
    return {
        name,
        size: formatBytes(sizeBytes),
        sizeBytes,
        format: extension,
        fileCount: files.length,
    };
}

function formatFileTypeLabel(fileName) {
    const lowerName = String(fileName || "").toLowerCase();
    if (!lowerName) {
        return "VOLUME";
    }
    if (lowerName.endsWith(".nii.gz")) {
        return "NII.GZ";
    }
    const segments = lowerName.split(".").filter(Boolean);
    if (segments.length <= 1) {
        return "VOLUME";
    }
    return segments[segments.length - 1].toUpperCase();
}

function formatDisplayFileName(fileName, maxLength = 56) {
    const value = String(fileName || "").trim();
    if (!value) {
        return "Unnamed file";
    }
    if (value.length <= maxLength) {
        return value;
    }
    const head = Math.ceil((maxLength - 1) * 0.6);
    const tail = Math.floor((maxLength - 1) * 0.4);
    return `${value.slice(0, head)}…${value.slice(-tail)}`;
}

function showUploadError(message) {
    uploadErrorBanner.hidden = false;
    uploadErrorBanner.textContent = message;
}

function hideUploadError() {
    uploadErrorBanner.hidden = true;
    uploadErrorBanner.textContent = "";
}

function startAnalysis(scan, onStart) {
    let progress = 12;
    let stepIndex = 0;

    analysisDetail.textContent = analysisMessages[stepIndex];
    analysisProgressBar.style.width = `${progress}%`;
    analysisProgressLabel.textContent = `${progress}% complete`;
    setActiveAnalysisStep(stepIndex);

    state.analysisStepTimer = window.setInterval(() => {
        stepIndex = (stepIndex + 1) % analysisMessages.length;
        analysisDetail.textContent = analysisMessages[stepIndex];
        setActiveAnalysisStep(stepIndex);
    }, 800);

    state.analysisTimer = window.setInterval(() => {
        progress = Math.min(progress + 9, 92);
        analysisProgressBar.style.width = `${progress}%`;
        analysisProgressLabel.textContent = `${progress}% complete`;
    }, 550);

    onStart();
}

function stopAnalysisAnimation() {
    if (state.analysisTimer) {
        window.clearInterval(state.analysisTimer);
        state.analysisTimer = null;
    }

    if (state.analysisStepTimer) {
        window.clearInterval(state.analysisStepTimer);
        state.analysisStepTimer = null;
    }
}

function setActiveAnalysisStep(stepIndex) {
    analysisSteps.forEach((step, index) => {
        step.classList.toggle("is-active", index === stepIndex);
    });
}

async function analyzeSelectedFiles() {
    const formData = new FormData();
    state.selectedFiles.forEach((file) => {
        formData.append("ct_scan", file);
    });

    const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
        throw new Error(payload.error || "Analysis failed.");
    }

    return payload;
}

function finalizeAnalysis(scan, payload) {
    stopAnalysisAnimation();
    analysisProgressBar.style.width = "100%";
    analysisProgressLabel.textContent = "100% complete";

    const enriched = {
        ...scan,
        status: payload.status || "Analysis complete",
        analyzedAt: new Date().toISOString(),
        result: payload,
    };

    state.currentScan = enriched;
    state.analysisResultById.set(scan.id, enriched);
    upsertHistory(enriched);
    renderRecentUploads();
    renderHistoryPage();

    window.setTimeout(() => {
        navigate(`/results/${scan.id}`);
    }, 300);
}

function renderResultsPage(scan) {
    const payload = scan.result;
    const analysis = payload.analysis || {};
    const overallRisk = String(analysis.overall_risk || "LOW").toUpperCase();
    const nodules = (analysis.nodules || []).filter((nodule) => {
        const filter = noduleFilter.value;
        return filter === "ALL" || String(nodule.risk_level || "LOW").toUpperCase() === filter;
    });

    riskBanner.classList.remove("risk-banner--low", "risk-banner--medium", "risk-banner--high");
    riskBanner.classList.add(`risk-banner--${overallRisk.toLowerCase()}`);
    riskIcon.textContent = overallRisk.charAt(0);
    riskLabel.textContent = `Patient Risk: ${overallRisk}`;
    riskSummary.textContent = `Maximum malignancy: ${Number(analysis.risk_score || 0).toFixed(1)}%`;
    riskRecommendation.textContent = payload.next_steps || "Consult physician for evaluation.";
    noduleCount.textContent = String(analysis.num_nodules_detected || 0);
    analysisDate.textContent = formatDate(scan.analyzedAt);
    analysisStatus.textContent = payload.status || "Analysis complete";
    resultsMeta.textContent = `${analysis.num_nodules_detected || 0} nodules detected • Analyzed ${formatDate(scan.analyzedAt)}`;
    resultsBreadcrumb.textContent = `Home > History > Scan ${scan.id}`;
    recommendationText.textContent = payload.next_steps || "Consult physician for evaluation.";
    scanFileName.textContent = formatDisplayFileName(scan.fileSummary?.name || "-");
    scanFileName.title = scan.fileSummary?.name || "-";
    scanFileSize.textContent = scan.fileSummary?.size || "-";
    scanFileFormat.textContent = scan.fileSummary?.format || "-";
    scanFileFormat.title = scan.fileSummary?.format || "-";

    renderTiming(payload.timing || {});
    renderVisualization(payload.visualization);
    renderNodules(nodules);
    updateResultNeighborButtons(scan.id);
}

function renderTiming(timing) {
    timingGrid.innerHTML = "";
    [
        ["Preprocess", timing.preprocess_sec],
        ["Detection", timing.detect_sec],
        ["Total", timing.total_sec],
    ].forEach(([label, value]) => {
        if (value == null) {
            return;
        }

        const item = document.createElement("div");
        item.innerHTML = `<span>${label}</span><strong>${Number(value).toFixed(1)}s</strong>`;
        timingGrid.appendChild(item);
    });
}

function renderVisualization(visualization) {
    if (visualization) {
        scanImage.src = `data:image/png;base64,${visualization}`;
        scanImage.hidden = false;
        scanPlaceholder.hidden = true;
        return;
    }

    scanImage.hidden = true;
    scanImage.removeAttribute("src");
    scanPlaceholder.hidden = false;
}

function renderNodules(nodules) {
    nodulesList.innerHTML = "";
    if (!nodules.length) {
        emptyState.hidden = false;
        return;
    }

    emptyState.hidden = true;

    nodules.forEach((nodule) => {
        const fragment = noduleCardTemplate.content.cloneNode(true);
        const riskLevel = String(nodule.risk_level || "LOW").toUpperCase();
        const detection = Number(nodule.detection_confidence || 0);
        const malignancy = Number(nodule.malignancy_probability || 0);

        fragment.querySelector("h3").textContent = `Nodule #${nodule.nodule_id}`;
        fragment.querySelector(".nodule-location").textContent = `Location: ${nodule.location}`;
        fragment.querySelector(".detection-value").textContent = `${detection.toFixed(1)}%`;
        fragment.querySelector(".malignancy-value").textContent = `${malignancy.toFixed(1)}%`;
        fragment.querySelector(".nodule-recommendation").textContent = nodule.recommendation || "Consult physician.";

        const pill = fragment.querySelector(".risk-pill");
        pill.textContent = riskLevel;
        pill.classList.add(`risk-pill--${riskLevel.toLowerCase()}`);

        const detectionBar = fragment.querySelector(".detection-bar");
        const malignancyBar = fragment.querySelector(".malignancy-bar");
        detectionBar.style.background = progressGradient(detection);
        malignancyBar.style.background = progressGradient(malignancy);

        nodulesList.appendChild(fragment);

        window.requestAnimationFrame(() => {
            detectionBar.style.width = `${detection}%`;
            malignancyBar.style.width = `${malignancy}%`;
        });
    });
}

function renderHistoryPage() {
    historyGroups.innerHTML = "";

    const searchValue = historySearch.value.trim().toLowerCase();
    const riskValue = historyFilter.value;
    const sortValue = historySort.value;

    const items = [...state.history]
        .filter((item) => {
            const matchesSearch =
                !searchValue ||
                item.fileSummary.name.toLowerCase().includes(searchValue) ||
                item.id.toLowerCase().includes(searchValue);
            const risk = String(item.result?.analysis?.overall_risk || "LOW").toUpperCase();
            const matchesRisk = riskValue === "ALL" || risk === riskValue;
            return matchesSearch && matchesRisk;
        })
        .sort((a, b) => {
            const aDate = new Date(a.analyzedAt || a.createdAt).getTime();
            const bDate = new Date(b.analyzedAt || b.createdAt).getTime();
            return sortValue === "OLDEST" ? aDate - bDate : bDate - aDate;
        });

    historySubtitle.textContent = `${items.length} scans analyzed`;

    if (!items.length) {
        historyGroups.innerHTML = `
            <article class="card">
                <h3>No scans yet</h3>
                <p class="summary-text">Analyze a CT scan to start building your history.</p>
            </article>
        `;
        return;
    }

    const grouped = groupByMonth(items);
    Object.entries(grouped).forEach(([month, scans]) => {
        const group = document.createElement("section");
        group.className = "history-group";
        const title = document.createElement("h2");
        title.textContent = month;
        group.appendChild(title);

        scans.forEach((scan) => {
            const fragment = historyCardTemplate.content.cloneNode(true);
            const risk = String(scan.result?.analysis?.overall_risk || "LOW").toUpperCase();
            const score = Number(scan.result?.analysis?.risk_score || 0);
            const bar = fragment.querySelector(".history-bar");
            bar.style.background = progressGradient(score);

            fragment.querySelector(".history-card__date").textContent = formatDate(scan.analyzedAt || scan.createdAt, true);
            fragment.querySelector("h3").textContent = `${scan.result?.analysis?.num_nodules_detected || 0} nodules detected`;
            fragment.querySelector(".history-card__score").textContent = `Maximum malignancy: ${score.toFixed(1)}%`;
            fragment.querySelector(".history-card__file").textContent = `${scan.fileSummary.name} • ${scan.fileSummary.size}`;
            bar.style.width = `${score}%`;

            const pill = fragment.querySelector(".risk-pill");
            pill.textContent = risk;
            pill.classList.add(`risk-pill--${risk.toLowerCase()}`);

            const viewLink = fragment.querySelector("[data-history-view]");
            viewLink.setAttribute("href", `/results/${scan.id}`);
            viewLink.addEventListener("click", (event) => {
                event.preventDefault();
                navigate(`/results/${scan.id}`);
            });

            fragment.querySelector("[data-history-export]").addEventListener("click", () => {
                state.currentScan = scan;
                openModal("export");
            });

            group.appendChild(fragment);
        });

        historyGroups.appendChild(group);
    });
}

function renderRecentUploads() {
    recentUploads.innerHTML = "";
    const items = state.history.slice(0, 3);
    if (!items.length) {
        recentUploads.innerHTML = `<p class="summary-text">No recent scans yet.</p>`;
        return;
    }

    items.forEach((item) => {
        const card = document.createElement("article");
        card.className = "recent-upload-item";
        card.innerHTML = `
            <strong>${item.fileSummary.name}</strong>
            <p>Uploaded ${timeAgo(item.analyzedAt || item.createdAt)} • ${item.fileSummary.size}</p>
        `;
        card.addEventListener("click", () => navigate(`/results/${item.id}`));
        recentUploads.appendChild(card);
    });
}

function resolveScan(scanId) {
    if (state.currentScan?.id === scanId) {
        return state.currentScan;
    }

    return state.history.find((item) => item.id === scanId) || null;
}

function upsertHistory(scan) {
    const existingIndex = state.history.findIndex((item) => item.id === scan.id);
    if (existingIndex >= 0) {
        state.history[existingIndex] = scan;
    } else {
        state.history.unshift(scan);
    }

    saveHistory(state.history);
}

function loadHistory() {
    try {
        return JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "[]");
    } catch {
        return [];
    }
}

function saveHistory(history) {
    const compactHistory = history
        .slice(0, 20)
        .map((scan) => ({
            ...scan,
            result: scan.result
                ? {
                    ...scan.result,
                    visualization: null,
                }
                : null,
        }));

    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(compactHistory));
    state.history = compactHistory;
}

function buildSampleScan(scanId) {
    const now = new Date().toISOString();
    return {
        id: scanId,
        createdAt: now,
        status: "Analyzing",
        fileSummary: {
            name: "sample_patient_scan.nii.gz",
            size: "128.4 MB",
            format: "NII.GZ",
            fileCount: 1,
        },
        result: {
            status: "Sample analysis complete",
            next_steps: "Review the highlighted nodule and consider follow-up imaging in 6 months.",
            analysis: {
                num_nodules_detected: 2,
                overall_risk: "MEDIUM",
                risk_score: 54.2,
                nodules: [
                    {
                        nodule_id: 1,
                        location: "(87, 40, 257)",
                        detection_confidence: 60.2,
                        malignancy_probability: 36.4,
                        risk_level: "MEDIUM",
                        recommendation: "Follow-up scan in 6 months.",
                    },
                    {
                        nodule_id: 2,
                        location: "(112, 63, 244)",
                        detection_confidence: 82.7,
                        malignancy_probability: 54.2,
                        risk_level: "HIGH",
                        recommendation: "Expedited physician review is advised.",
                    },
                ],
            },
            visualization: null,
            timing: {
                preprocess_sec: 7.2,
                detect_sec: 18.4,
                total_sec: 28.6,
            },
        },
    };
}

function exportCurrentImage() {
    if (scanImage.src) {
        const link = document.createElement("a");
        link.href = scanImage.src;
        link.download = "oncovision-ct-visualization.png";
        link.click();
        return;
    }

    toastLikeAlert("No CT visualization is available for export yet.");
}

function navigateHistoryNeighbor(direction) {
    if (!state.currentScan) {
        return;
    }

    const index = state.history.findIndex((item) => item.id === state.currentScan.id);
    if (index < 0) {
        return;
    }

    const next = state.history[index + direction];
    if (!next) {
        return;
    }

    navigate(`/results/${next.id}`);
}

function updateResultNeighborButtons(scanId) {
    const index = state.history.findIndex((item) => item.id === scanId);
    previousScanButton.disabled = index <= 0;
    nextScanButton.disabled = index < 0 || index >= state.history.length - 1;
}

function deleteCurrentScan() {
    if (!state.currentScan) {
        return;
    }

    const confirmed = window.confirm(`Delete ${state.currentScan.fileSummary?.name || state.currentScan.id} from local history?`);
    if (!confirmed) {
        return;
    }

    state.history = state.history.filter((item) => item.id !== state.currentScan.id);
    saveHistory(state.history);
    renderHistoryPage();
    renderRecentUploads();
    state.currentScan = null;
    navigate("/history");
}

function confirmShare() {
    const email = shareEmailInput.value.trim();
    if (!email) {
        toastLikeAlert("Enter an email address to share results.");
        return;
    }

    const shareUrl = `${window.location.origin}/results/${state.currentScan?.id || ""}`;
    const message = shareMessageInput.value.trim() || "Please review this scan.";
    openModal(null);
    toastLikeAlert(`Prepared share for ${email}\n\nLink: ${shareUrl}\n\nMessage: ${message}`);
}

function generatePdfReport(previewOnly) {
    if (!state.currentScan?.result) {
        toastLikeAlert("No scan result is available to export.");
        return;
    }

    const scan = state.currentScan;
    const analysis = scan.result.analysis || {};
    const reportData = {
        patientName: patientNameInput.value.trim() || "Not provided",
        mrn: patientMrnInput.value.trim() || "Not provided",
        dob: patientDobInput.value.trim() || "Not provided",
        physicianNotes: physicianNotesInput.value.trim() || "No additional physician notes.",
        nodules: analysis.nodules || [],
        overallRisk: String(analysis.overall_risk || "LOW").toUpperCase(),
        riskScore: Number(analysis.risk_score || 0).toFixed(1),
        totalTime: Number(scan.result.timing?.total_sec || 0).toFixed(1),
        scanDate: formatDate(scan.analyzedAt || scan.createdAt, true),
        generatedAt: formatDate(new Date().toISOString(), true),
        fileName: scan.fileSummary?.name || "-",
        fileSize: scan.fileSummary?.size || "-",
        nextSteps: scan.result.next_steps || "Consult physician for evaluation.",
        clinicalHighlights: buildClinicalHighlights(analysis.nodules || [], scan.result.next_steps || "Consult physician for evaluation."),
        visualizationSrc: scanImage.src || "",
        riskColor: riskBanner.classList.contains("risk-banner--high")
            ? "#fee2e2"
            : riskBanner.classList.contains("risk-banner--medium")
                ? "#fef3c7"
                : "#d1fae5",
    };
    const reportHtml = buildPdfPreviewHtml(scan, reportData);

    if (previewOnly) {
        const reportWindow = window.open("", "_blank", "noopener,noreferrer,width=1100,height=900");
        if (!reportWindow) {
            toastLikeAlert("Popup blocked. Allow popups to preview the report.");
            return;
        }

        reportWindow.document.open();
        reportWindow.document.write(reportHtml);
        reportWindow.document.close();
        openModal(null);
        return;
    }

    const jsPdfCtor = window.jspdf?.jsPDF || window.jsPDF;
    if (!jsPdfCtor) {
        toastLikeAlert("PDF library failed to load. Falling back to preview window.");
        generatePdfReport(true);
        return;
    }

    const filename = `OVX_Report_${scan.id}_${formatReportDate(new Date().toISOString())}.pdf`;
    openModal(null);

    createPdfDocument(scan, reportData, jsPdfCtor, filename).catch((error) => {
        console.error("PDF generation failed", error);
        toastLikeAlert("PDF generation failed. Opening preview instead.");
        generatePdfReport(true);
    });
}

function buildPdfPreviewHtml(scan, reportData) {
    const visualization = reportData.visualizationSrc
        ? `<img src="${reportData.visualizationSrc}" alt="CT visualization" style="width:100%;height:auto;border:1px solid #d6dee8;border-radius:18px;">`
        : "<div style='padding:32px;border:1px solid #d6dee8;border-radius:18px;background:#f8fafc;'>Visualization not available for this scan.</div>";
    const nodulesHtml = reportData.nodules.length
        ? reportData.nodules.map((nodule) => `<div class="box" style="margin-bottom:12px"><h3>Nodule #${nodule.nodule_id}</h3><p>Risk: ${escapeHtml(nodule.risk_level || "LOW")}</p><p>Detection Confidence: ${Number(nodule.detection_confidence || 0).toFixed(1)}%</p><p>Malignancy Probability: ${Number(nodule.malignancy_probability || 0).toFixed(1)}%</p><p>Location: ${escapeHtml(nodule.location || "-")}</p><p>Recommendation: ${escapeHtml(nodule.recommendation || "Consult physician.")}</p></div>`).join("")
        : "<p>No suspicious nodules were detected.</p>";

    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>OVX Report ${scan.id}</title>
<style>
body{font-family:Inter,Arial,sans-serif;margin:0;color:#1f2937;background:#eef2f7}
.page{box-sizing:border-box;padding:22mm 18mm;min-height:297mm;page-break-after:always;position:relative;background:#fff}
.page:last-child{page-break-after:auto}
h1,h2,h3{color:#0a1628;margin:0 0 12px}
h1{font-size:30px} h2{font-size:24px} p,li{font-size:13px;line-height:1.6}
.brand{width:220px;margin-bottom:28px}
.risk{padding:16px;border-radius:10px;background:${reportData.riskColor};margin:18px 0}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}
.box{border:1px solid #e5e7eb;border-radius:10px;padding:14px;background:#fff}
.footer{position:absolute;left:20mm;right:20mm;bottom:12mm;padding-top:6mm;border-top:1px solid #e5e7eb;font-size:10px;color:#6b7280;display:flex;justify-content:space-between}
.watermark{position:absolute;inset:0;display:grid;place-items:center;pointer-events:none}
.watermark img{width:320px;opacity:.055;filter:grayscale(1)}
.visual-shell{margin-top:14px;padding:14px;border:1px solid #d6dee8;border-radius:20px;background:linear-gradient(180deg,#f8fafc 0%,#eef4fb 100%)}
.visual-caption{margin-top:12px;font-size:12px;color:#475569}
</style>
</head>
<body>
<section class="page">
<div class="watermark"><img src="${window.location.origin}/assets/favicon.png" alt=""></div>
<img class="brand" src="${window.location.origin}/assets/logo.png" alt="OncoVision-X">
<h1>Lung Cancer Screening Analysis Report</h1>
<p>Scan ID: ${scan.id}</p>
<p>Scan Date: ${reportData.scanDate}</p>
<p>Report Generated: ${reportData.generatedAt}</p>
<div class="risk"><strong>Overall Risk: ${reportData.overallRisk}</strong><br>Maximum malignancy probability: ${reportData.riskScore}%</div>
<p>${escapeHtml(reportData.nextSteps)}</p>
<div class="box"><h3>Clinical Action Summary</h3>${reportData.clinicalHighlights.map((item) => `<p style="margin:0 0 8px">• ${escapeHtml(item)}</p>`).join("")}</div>
<div class="grid">
<div class="box"><strong>Patient</strong><p>${escapeHtml(reportData.patientName)}</p></div>
<div class="box"><strong>MRN</strong><p>${escapeHtml(reportData.mrn)}</p></div>
<div class="box"><strong>DOB</strong><p>${escapeHtml(reportData.dob)}</p></div>
<div class="box"><strong>File</strong><p>${escapeHtml(reportData.fileName)}</p></div>
<div class="box"><strong>Size</strong><p>${escapeHtml(reportData.fileSize)}</p></div>
<div class="box"><strong>Total Time</strong><p>${reportData.totalTime}s</p></div>
</div>
<div class="footer"><span>OncoVision-X • Confidential Report</span><span>Page 1 of 3</span></div>
</section>
<section class="page">
<div class="watermark"><img src="${window.location.origin}/assets/favicon.png" alt=""></div>
<h2>Detailed Findings</h2>
${nodulesHtml}
<div class="box"><h3>Physician Notes</h3><p>${escapeHtml(reportData.physicianNotes)}</p></div>
<div class="footer"><span>OncoVision-X • Professional Report</span><span>Page 2 of 3</span></div>
</section>
<section class="page">
<div class="watermark"><img src="${window.location.origin}/assets/favicon.png" alt=""></div>
<h2>Three-View Scan Review</h2>
<div class="visual-shell">
${visualization}
</div>
<p class="visual-caption">The axial, coronal, and sagittal CT views are preserved together on a single report page for review and sharing.</p>
<p style="margin-top:18px">This report is intended to support clinical review and does not replace professional medical judgment. All findings should be reviewed by a qualified physician or radiologist.</p>
<div class="footer"><span>OncoVision-X • Professional Report</span><span>Page 3 of 3</span></div>
</section>
</body>
</html>`;
}

async function createPdfDocument(scan, reportData, JsPdfCtor, filename) {
    const doc = new JsPdfCtor({
        unit: "mm",
        format: "a4",
        orientation: "portrait",
    });
    const margin = 16;
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const contentWidth = pageWidth - margin * 2;
    const [logoSrc, watermarkSrc, visualizationSrc, visualizationSize, logoSize] = await Promise.all([
        imageToDataUrl("/assets/logo.png"),
        imageToDataUrl("/assets/favicon.png"),
        reportData.visualizationSrc ? imageToDataUrl(reportData.visualizationSrc) : Promise.resolve(""),
        reportData.visualizationSrc ? getImageSize(reportData.visualizationSrc) : Promise.resolve(null),
        getImageSize("/assets/logo.png"),
    ]);
    const viewImages = visualizationSrc ? await splitThreeViewVisualization(visualizationSrc) : [];
    const detailedFindingsPages = estimateDetailedFindingsPages(
        doc,
        contentWidth,
        reportData.nodules,
        reportData.physicianNotes,
        pageHeight,
    );
    const totalPages = 1 + detailedFindingsPages + Math.max(viewImages.length, 1);

    drawPdfPageHeader(doc, {
        title: "Lung Screening Imaging Report",
        subtitleLines: [
            `Scan ID: ${scan.id}`,
            `Scan Date: ${reportData.scanDate}`,
            `Report Generated: ${reportData.generatedAt}`,
            "Prepared for clinical review and print distribution",
        ],
        logoSrc,
        logoSize,
        watermarkSrc,
    });
    drawRiskBanner(doc, margin, 72, contentWidth, reportData);
    let y = drawWrappedTextBlock(doc, reportData.nextSteps, margin, 98, contentWidth, {
        fontSize: 11,
        textColor: "#334155",
        lineHeight: 5.8,
    }) + 8;
    y = drawHighlightsBlock(doc, margin, y, contentWidth, reportData.clinicalHighlights) + 6;
    drawInfoGrid(doc, margin, y, contentWidth, [
        ["Patient", reportData.patientName],
        ["MRN", reportData.mrn],
        ["DOB", reportData.dob],
        ["File", reportData.fileName],
        ["Size", reportData.fileSize],
        ["Total Time", `${reportData.totalTime}s`],
    ]);
    drawPdfFooter(doc, 1, totalPages);

    doc.addPage();
    let currentPageNumber = 2;
    drawPdfPageHeader(doc, {
        title: "Detailed Findings",
        subtitleLines: [],
        logoSrc,
        logoSize,
        watermarkSrc,
    });
    y = 64;
    if (reportData.nodules.length) {
        reportData.nodules.forEach((nodule, index) => {
            const blockHeight = getNoduleBlockHeight(doc, contentWidth, nodule);
            if (y + blockHeight > pageHeight - 26) {
                drawPdfFooter(doc, currentPageNumber, totalPages);
                doc.addPage();
                currentPageNumber += 1;
                drawPdfPageHeader(doc, {
                    title: "Detailed Findings",
                    subtitleLines: ["Continued findings for clinical review"],
                    logoSrc,
                    logoSize,
                    watermarkSrc,
                });
                y = 64;
            }
            drawNoduleBlock(doc, margin, y, contentWidth, index + 1, nodule);
            y += blockHeight + 4;
        });
    } else {
        y = drawWrappedTextBlock(doc, "No suspicious nodules were detected.", margin, y, contentWidth, {
            fontSize: 11,
            textColor: "#475569",
            lineHeight: 5.8,
        }) + 8;
    }
    const notesHeight = getNotesBlockHeight(doc, contentWidth, reportData.physicianNotes);
    if (y + notesHeight > pageHeight - 26) {
        drawPdfFooter(doc, currentPageNumber, totalPages);
        doc.addPage();
        currentPageNumber += 1;
        drawPdfPageHeader(doc, {
            title: "Physician Notes",
            subtitleLines: ["Continued findings and clinician comments"],
            logoSrc,
            logoSize,
            watermarkSrc,
        });
        y = 64;
    }
    drawNotesBlock(doc, margin, y, contentWidth, reportData.physicianNotes);
    drawPdfFooter(doc, currentPageNumber, totalPages);

    const labeledViews = viewImages.length
        ? viewImages.map((view, index) => ({
            ...view,
            label: ["Axial View", "Coronal View", "Sagittal View"][index] || `View ${index + 1}`,
        }))
        : [{ src: visualizationSrc, label: "Scan View", width: 0, height: 0 }];

    labeledViews.forEach((viewImage, index) => {
        doc.addPage();
        drawPdfPageHeader(doc, {
            title: viewImage.label,
            subtitleLines: [
                "High-resolution imaging panel for clinical review",
                `Scan ID: ${scan.id}`,
            ],
            logoSrc,
            logoSize,
            watermarkSrc,
        });
        drawSingleViewPage(doc, {
            x: margin,
            y: 58,
            width: contentWidth,
            height: 172,
            viewImage,
        });
        drawWrappedTextBlock(
            doc,
            "For interpretation and print review by qualified clinical staff.",
            margin,
            240,
            contentWidth,
            {
                fontSize: 9,
                textColor: "#64748b",
                lineHeight: 4.6,
            },
        );
        drawPdfFooter(doc, currentPageNumber + 1 + index, totalPages);
    });
    doc.save(filename);
    toastLikeAlert("Report downloaded successfully.");
}

function drawPdfPageHeader(doc, { title, subtitleLines, logoSrc, logoSize, watermarkSrc }) {
    if (watermarkSrc) {
        drawPdfWatermark(doc, watermarkSrc);
    }
    if (logoSrc) {
        const fittedLogo = fitImageInBox(
            logoSize?.width || 540,
            logoSize?.height || 140,
            68,
            22,
        );
        doc.addImage(logoSrc, "PNG", 16, 12, fittedLogo.width, fittedLogo.height);
    }
    doc.setFont("helvetica", "bold");
    doc.setFontSize(19);
    doc.setTextColor("#0f172a");
    doc.text(title, 16, 39);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.setTextColor("#475569");
    subtitleLines.forEach((line, index) => {
        doc.text(line, 16, 47 + index * 4.8);
    });
}

function drawPdfWatermark(doc, watermarkSrc) {
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    if (typeof doc.GState === "function" && typeof doc.setGState === "function") {
        doc.setGState(new doc.GState({ opacity: 0.06 }));
        doc.addImage(watermarkSrc, "PNG", pageWidth / 2 - 52, pageHeight / 2 - 52, 104, 104);
        doc.setGState(new doc.GState({ opacity: 1 }));
        return;
    }
    doc.addImage(watermarkSrc, "PNG", pageWidth / 2 - 44, pageHeight / 2 - 44, 88, 88);
}

function drawRiskBanner(doc, x, y, width, reportData) {
    const rgb = hexToRgb(reportData.riskColor);
    doc.setFillColor(rgb.r, rgb.g, rgb.b);
    doc.roundedRect(x, y, width, 18, 4, 4, "F");
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.setTextColor("#0f172a");
    doc.text(`Overall Risk: ${reportData.overallRisk}`, x + 4, y + 7);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10.5);
    doc.text(`Maximum malignancy probability: ${reportData.riskScore}%`, x + 4, y + 13);
}

function drawInfoGrid(doc, x, y, width, items) {
    const columns = 2;
    const gap = 4;
    const boxWidth = (width - gap) / columns;
    const boxHeight = 18;
    items.forEach(([label, value], index) => {
        const col = index % columns;
        const row = Math.floor(index / columns);
        const boxX = x + col * (boxWidth + gap);
        const boxY = y + row * (boxHeight + gap);
        doc.setDrawColor("#d7e0ea");
        doc.setFillColor(255, 255, 255);
        doc.roundedRect(boxX, boxY, boxWidth, boxHeight, 3, 3, "FD");
        doc.setFont("helvetica", "bold");
        doc.setFontSize(9.5);
        doc.setTextColor("#0f172a");
        doc.text(label, boxX + 3, boxY + 6);
        doc.setFont("helvetica", "normal");
        doc.setFontSize(10);
        doc.setTextColor("#475569");
        const lines = doc.splitTextToSize(String(value), boxWidth - 6);
        doc.text(lines[0] || "-", boxX + 3, boxY + 12);
    });
}

function drawHighlightsBlock(doc, x, y, width, items) {
    const lineCounts = items.map((item) => doc.splitTextToSize(`• ${String(item)}`, width - 8).length);
    const blockHeight = 11 + lineCounts.reduce((sum, count) => sum + count * 5, 0);
    doc.setDrawColor("#d7e0ea");
    doc.setFillColor(255, 255, 255);
    doc.roundedRect(x, y, width, blockHeight, 3, 3, "FD");
    doc.setFont("helvetica", "bold");
    doc.setFontSize(11);
    doc.setTextColor("#0f172a");
    doc.text("Clinical Action Summary", x + 4, y + 7);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9.5);
    doc.setTextColor("#475569");
    let cursorY = y + 13;
    items.forEach((item, index) => {
        const lines = doc.splitTextToSize(`• ${String(item)}`, width - 8);
        doc.text(lines, x + 4, cursorY);
        cursorY += lines.length * 5;
    });
    return y + blockHeight;
}

function getNoduleBlockHeight(doc, width, nodule) {
    const location = doc.splitTextToSize(`Location: ${String(nodule.location || "-")}`, width - 114);
    const recommendation = doc.splitTextToSize(`Recommendation: ${String(nodule.recommendation || "Consult physician.")}`, width - 114);
    const rightColumnLines = location.length + recommendation.length;
    return Math.max(54, 14 + rightColumnLines * 5.2);
}

function estimateDetailedFindingsPages(doc, width, nodules, notes, pageHeight) {
    let pages = 1;
    let y = 40;
    if (nodules.length) {
        nodules.forEach((nodule) => {
            const blockHeight = getNoduleBlockHeight(doc, width, nodule);
            if (y + blockHeight > pageHeight - 26) {
                pages += 1;
                y = 40;
            }
            y += blockHeight + 4;
        });
    } else {
        y += 14;
    }
    const notesHeight = getNotesBlockHeight(doc, width, notes);
    if (y + notesHeight > pageHeight - 26) {
        pages += 1;
    }
    return pages;
}

function drawNoduleBlock(doc, x, y, width, ordinal, nodule) {
    const blockHeight = getNoduleBlockHeight(doc, width, nodule);
    doc.setDrawColor("#d7e0ea");
    doc.setFillColor(255, 255, 255);
    doc.roundedRect(x, y, width, blockHeight, 3, 3, "FD");
    doc.setFont("helvetica", "bold");
    doc.setFontSize(11);
    doc.setTextColor("#0f172a");
    doc.text(`Nodule #${nodule.nodule_id || ordinal}`, x + 4, y + 7);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9.5);
    doc.setTextColor("#334155");
    doc.text(`Risk: ${String(nodule.risk_level || "LOW").toUpperCase()}`, x + 4, y + 14);
    doc.text(`Detection Confidence: ${Number(nodule.detection_confidence || 0).toFixed(1)}%`, x + 4, y + 20);
    doc.text(`Malignancy Probability: ${Number(nodule.malignancy_probability || 0).toFixed(1)}%`, x + 4, y + 26);
    const location = doc.splitTextToSize(`Location: ${String(nodule.location || "-")}`, width - 114);
    const recommendation = doc.splitTextToSize(`Recommendation: ${String(nodule.recommendation || "Consult physician.")}`, width - 114);
    doc.text(location, x + 110, y + 14);
    doc.text(recommendation, x + 110, y + 14 + location.length * 5.2 + 2);
}

function getNotesBlockHeight(doc, width, notes) {
    const lines = doc.splitTextToSize(String(notes || ""), width - 8);
    return Math.max(38, 16 + lines.length * 5);
}

function drawNotesBlock(doc, x, y, width, notes) {
    const blockHeight = getNotesBlockHeight(doc, width, notes);
    doc.setDrawColor("#d7e0ea");
    doc.setFillColor(255, 255, 255);
    doc.roundedRect(x, y, width, blockHeight, 3, 3, "FD");
    doc.setFont("helvetica", "bold");
    doc.setFontSize(11);
    doc.setTextColor("#0f172a");
    doc.text("Physician Notes", x + 4, y + 7);
    drawWrappedTextBlock(doc, notes, x + 4, y + 14, width - 8, {
        fontSize: 9.5,
        textColor: "#475569",
        lineHeight: 5,
    });
}

function drawSingleViewPage(doc, { x, y, width, height, viewImage }) {
    doc.setDrawColor("#d7e0ea");
    doc.setFillColor(248, 250, 252);
    doc.roundedRect(x, y, width, height, 6, 6, "FD");
    if (!viewImage?.src) {
        doc.setFont("helvetica", "normal");
        doc.setFontSize(11);
        doc.setTextColor("#64748b");
        doc.text("Visualization not available for this scan.", x + width / 2, y + height / 2, { align: "center" });
        return;
    }
    const fitted = fitImageInBox(
        viewImage.width || 1,
        viewImage.height || 1,
        width - 12,
        height - 12,
    );
    doc.addImage(
        viewImage.src,
        "PNG",
        x + 6 + (width - 12 - fitted.width) / 2,
        y + 6 + (height - 12 - fitted.height) / 2,
        fitted.width,
        fitted.height,
    );
}

function drawPdfFooter(doc, pageNumber, totalPages) {
    const pageHeight = doc.internal.pageSize.getHeight();
    doc.setDrawColor("#d7e0ea");
    doc.line(16, pageHeight - 14, 194, pageHeight - 14);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(8.5);
    doc.setTextColor("#64748b");
    doc.text("OncoVision-X • Professional Report", 16, pageHeight - 8);
    doc.text(pageNumberLabel(pageNumber, totalPages), 194, pageHeight - 8, { align: "right" });
}

function drawWrappedTextBlock(doc, text, x, y, width, options = {}) {
    const {
        fontSize = 10.5,
        textColor = "#334155",
        lineHeight = 5,
    } = options;
    doc.setFont("helvetica", "normal");
    doc.setFontSize(fontSize);
    doc.setTextColor(textColor);
    const lines = doc.splitTextToSize(String(text || ""), width);
    doc.text(lines, x, y);
    return y + lines.length * lineHeight;
}

function pageNumberLabel(pageNumber, totalPages) {
    return `Page ${pageNumber} of ${totalPages}`;
}

function buildClinicalHighlights(nodules, nextSteps) {
    const items = [];
    if (nextSteps) {
        items.push(nextSteps);
    }
    if (!nodules.length) {
        items.push("No suspicious nodules were identified on the current review.");
        return items;
    }
    const highRisk = nodules.filter((nodule) => String(nodule.risk_level || "").toUpperCase() === "HIGH").length;
    const largestProbability = Math.max(...nodules.map((nodule) => Number(nodule.malignancy_probability || 0)));
    items.push(`${nodules.length} finding${nodules.length > 1 ? "s are" : " is"} listed in this report for focused review.`);
    if (highRisk > 0) {
        items.push(`${highRisk} finding${highRisk > 1 ? "s are" : " is"} categorized as high risk and should be prioritized during review.`);
    } else if (largestProbability >= 45) {
        items.push("Intermediate-risk imaging features are present, so interval comparison is recommended.");
    } else {
        items.push("The visible findings are lower risk and are best managed with surveillance and future comparison.");
    }
    items.push("Compare the listed lesion locations with prior chest imaging whenever earlier studies are available.");
    return items.slice(0, 4);
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function imageToDataUrl(src) {
    if (!src) {
        return Promise.resolve("");
    }
    if (src.startsWith("data:")) {
        return Promise.resolve(src);
    }
    return fetch(src)
        .then((response) => {
            if (!response.ok) {
                throw new Error(`Failed to load image: ${src}`);
            }
            return response.blob();
        })
        .then((blob) => new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(String(reader.result || ""));
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        }));
}

function getImageSize(src) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight });
        image.onerror = reject;
        image.src = src;
    });
}

function splitThreeViewVisualization(src) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => {
            const segmentWidth = Math.floor(image.naturalWidth / 3);
            const views = [];
            for (let index = 0; index < 3; index += 1) {
                const canvas = document.createElement("canvas");
                const cropWidth = index === 2 ? image.naturalWidth - segmentWidth * 2 : segmentWidth;
                canvas.width = cropWidth;
                canvas.height = image.naturalHeight;
                const context = canvas.getContext("2d");
                if (!context) {
                    reject(new Error("Canvas context unavailable for visualization split"));
                    return;
                }
                context.drawImage(
                    image,
                    index * segmentWidth,
                    0,
                    cropWidth,
                    image.naturalHeight,
                    0,
                    0,
                    cropWidth,
                    image.naturalHeight,
                );
                views.push({
                    src: canvas.toDataURL("image/png"),
                    width: cropWidth,
                    height: image.naturalHeight,
                });
            }
            resolve(views);
        };
        image.onerror = reject;
        image.src = src;
    });
}

function fitImageInBox(sourceWidth, sourceHeight, maxWidth, maxHeight) {
    if (!sourceWidth || !sourceHeight) {
        return { width: maxWidth, height: maxHeight };
    }
    const scale = Math.min(maxWidth / sourceWidth, maxHeight / sourceHeight);
    return {
        width: sourceWidth * scale,
        height: sourceHeight * scale,
    };
}

function hexToRgb(hex) {
    const value = hex.replace("#", "");
    return {
        r: parseInt(value.slice(0, 2), 16),
        g: parseInt(value.slice(2, 4), 16),
        b: parseInt(value.slice(4, 6), 16),
    };
}

function openModal(type) {
    settingsModal.hidden = true;
    helpModal.hidden = true;
    exportModal.hidden = true;
    shareModal.hidden = true;
    modalBackdrop.hidden = type == null;
    state.activeModal = type;

    if (type === "settings") {
        settingsModal.hidden = false;
    } else if (type === "help") {
        helpModal.hidden = false;
    } else if (type === "export") {
        exportModal.hidden = false;
    } else if (type === "share") {
        shareModal.hidden = false;
    }
}

function groupByMonth(items) {
    return items.reduce((groups, item) => {
        const key = new Intl.DateTimeFormat("en-US", {
            month: "long",
            year: "numeric",
        }).format(new Date(item.analyzedAt || item.createdAt));
        groups[key] = groups[key] || [];
        groups[key].push(item);
        return groups;
    }, {});
}

function progressGradient(value) {
    if (value < 30) {
        return "linear-gradient(90deg, #34d399 0%, #10b981 100%)";
    }

    if (value < 70) {
        return "linear-gradient(90deg, #fbbf24 0%, #f59e0b 100%)";
    }

    return "linear-gradient(90deg, #f87171 0%, #ef4444 100%)";
}

function createScanId() {
    return `scan-${Date.now().toString(36)}`;
}

function formatBytes(bytes) {
    if (!bytes) {
        return "0 B";
    }

    const units = ["B", "KB", "MB", "GB"];
    const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    const value = bytes / 1024 ** index;
    return `${value.toFixed(value >= 100 || index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatDate(value, withTime = false) {
    if (!value) {
        return "-";
    }

    return new Intl.DateTimeFormat("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
        ...(withTime ? { hour: "numeric", minute: "2-digit" } : {}),
    }).format(new Date(value));
}

function formatReportDate(value) {
    const date = new Date(value);
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    return `${year}${month}${day}`;
}

function timeAgo(value) {
    const diff = Date.now() - new Date(value).getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    if (days <= 0) {
        return "today";
    }
    if (days === 1) {
        return "1 day ago";
    }
    return `${days} days ago`;
}

function toastLikeAlert(message) {
    window.alert(message);
}
