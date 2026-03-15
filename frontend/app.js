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
    selectedFileName.textContent = summary.name;
    selectedFileMeta.textContent = `${summary.size} • ${summary.format}`;
    fileFormatValue.textContent = summary.format;
    fileCountValue.textContent = String(summary.fileCount);
    fileSizeValue.textContent = summary.size;
}

function createFileSummary(files) {
    const name = files.length === 1 ? files[0].name : `${files.length} files selected`;
    const sizeBytes = files.reduce((sum, file) => sum + file.size, 0);
    const extension = files[0]?.name.split(".").slice(1).join(".") || "volume";
    return {
        name,
        size: formatBytes(sizeBytes),
        sizeBytes,
        format: extension.toUpperCase(),
        fileCount: files.length,
    };
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
    scanFileName.textContent = scan.fileSummary?.name || "-";
    scanFileSize.textContent = scan.fileSummary?.size || "-";
    scanFileFormat.textContent = scan.fileSummary?.format || "-";

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
            <article class="card" style="max-width: var(--content-width);">
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
    const patientName = patientNameInput.value.trim() || "Not provided";
    const mrn = patientMrnInput.value.trim() || "Not provided";
    const dob = patientDobInput.value.trim() || "Not provided";
    const physicianNotes = physicianNotesInput.value.trim() || "No additional physician notes.";
    const nodules = analysis.nodules || [];
    const visualization = scanImage.src
        ? `<img src="${scanImage.src}" alt="CT visualization" style="width:100%;border:1px solid #e5e7eb;border-radius:12px;">`
        : "<div style='padding:32px;border:1px solid #e5e7eb;border-radius:12px;background:#f8fafc;'>Visualization not available for this scan.</div>";
    const reportHtml = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>OVX Report ${scan.id}</title>
<style>
body{font-family:Inter,Arial,sans-serif;margin:0;color:#1f2937;background:#fff}
.page{padding:28mm 20mm;min-height:297mm;page-break-after:always;position:relative}
.page:last-child{page-break-after:auto}
h1,h2,h3{color:#0a1628;margin:0 0 12px}
h1{font-size:32px} h2{font-size:24px} p,li{font-size:13px;line-height:1.6}
.brand{width:220px;margin-bottom:36px}
.risk{padding:16px;border-radius:10px;background:${riskBanner.classList.contains("risk-banner--high") ? "#fee2e2" : riskBanner.classList.contains("risk-banner--medium") ? "#fef3c7" : "#d1fae5"};margin:18px 0}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}
.box{border:1px solid #e5e7eb;border-radius:10px;padding:14px;background:#fff}
.footer{position:absolute;left:20mm;right:20mm;bottom:12mm;padding-top:6mm;border-top:1px solid #e5e7eb;font-size:10px;color:#6b7280;display:flex;justify-content:space-between}
.watermark{position:absolute;inset:0;display:grid;place-items:center;pointer-events:none}
.watermark img{width:320px;opacity:.055;filter:grayscale(1)}
</style>
</head>
<body>
<section class="page">
<div class="watermark"><img src="${window.location.origin}/assets/favicon.png" alt=""></div>
<img class="brand" src="${window.location.origin}/assets/logo.png" alt="OncoVision-X">
<h1>Lung Cancer Screening Analysis Report</h1>
<p>Scan ID: ${scan.id}</p>
<p>Scan Date: ${formatDate(scan.analyzedAt || scan.createdAt, true)}</p>
<p>Report Generated: ${formatDate(new Date().toISOString(), true)}</p>
<div class="footer"><span>OncoVision-X • Confidential Report</span><span>Page 1 of 4</span></div>
</section>
<section class="page">
<div class="watermark"><img src="${window.location.origin}/assets/favicon.png" alt=""></div>
<h2>Executive Summary</h2>
<div class="risk"><strong>Overall Risk: ${analysis.overall_risk || "LOW"}</strong><br>Maximum malignancy probability: ${Number(analysis.risk_score || 0).toFixed(1)}%</div>
<p>${scan.result.next_steps || "Consult physician for evaluation."}</p>
<div class="grid">
<div class="box"><strong>Patient</strong><p>${patientName}</p></div>
<div class="box"><strong>MRN</strong><p>${mrn}</p></div>
<div class="box"><strong>DOB</strong><p>${dob}</p></div>
<div class="box"><strong>File</strong><p>${scan.fileSummary?.name || "-"}</p></div>
<div class="box"><strong>Size</strong><p>${scan.fileSummary?.size || "-"}</p></div>
<div class="box"><strong>Total Time</strong><p>${Number(scan.result.timing?.total_sec || 0).toFixed(1)}s</p></div>
</div>
<div class="footer"><span>OncoVision-X • Professional Report</span><span>Page 2 of 4</span></div>
</section>
<section class="page">
<div class="watermark"><img src="${window.location.origin}/assets/favicon.png" alt=""></div>
<h2>Detailed Findings</h2>
${nodules.length ? nodules.map((nodule) => `<div class="box" style="margin-bottom:12px"><h3>Nodule #${nodule.nodule_id}</h3><p>Risk: ${nodule.risk_level}</p><p>Detection Confidence: ${nodule.detection_confidence}%</p><p>Malignancy Probability: ${nodule.malignancy_probability}%</p><p>Location: ${nodule.location}</p><p>Recommendation: ${nodule.recommendation}</p></div>`).join("") : "<p>No suspicious nodules were detected.</p>"}
<div class="box"><h3>Physician Notes</h3><p>${physicianNotes}</p></div>
<div class="footer"><span>OncoVision-X • Professional Report</span><span>Page 3 of 4</span></div>
</section>
<section class="page">
<div class="watermark"><img src="${window.location.origin}/assets/favicon.png" alt=""></div>
<h2>Visualization & Disclaimer</h2>
${visualization}
<p style="margin-top:18px">This report is AI-assisted clinical decision support and does not replace professional medical judgment. All findings should be reviewed by a qualified physician or radiologist.</p>
<div class="footer"><span>OncoVision-X • Professional Report</span><span>Page 4 of 4</span></div>
</section>
</body>
</html>`;

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

    const html2pdfInstance = window.html2pdf;
    if (!html2pdfInstance) {
        toastLikeAlert("PDF library failed to load. Falling back to preview window.");
        generatePdfReport(true);
        return;
    }

    const styleMatch = reportHtml.match(/<style>([\s\S]*?)<\/style>/i);
    const bodyMatch = reportHtml.match(/<body>([\s\S]*?)<\/body>/i);
    const reportRoot = document.createElement("div");
    reportRoot.style.position = "fixed";
    reportRoot.style.left = "-99999px";
    reportRoot.style.top = "0";
    reportRoot.style.width = "210mm";
    reportRoot.style.background = "#ffffff";
    reportRoot.innerHTML = `${styleMatch ? `<style>${styleMatch[1]}</style>` : ""}${bodyMatch ? bodyMatch[1] : ""}`;
    document.body.appendChild(reportRoot);

    const filename = `OVX_Report_${scan.id}_${formatReportDate(new Date().toISOString())}.pdf`;
    openModal(null);

    html2pdfInstance()
        .set({
            margin: 0,
            filename,
            image: { type: "jpeg", quality: 0.98 },
            html2canvas: {
                scale: 2,
                useCORS: true,
                backgroundColor: "#ffffff",
            },
            jsPDF: {
                unit: "mm",
                format: "a4",
                orientation: "portrait",
            },
            pagebreak: { mode: ["css", "legacy"] },
        })
        .from(reportRoot)
        .save()
        .then(() => {
            document.body.removeChild(reportRoot);
            toastLikeAlert("Report downloaded successfully.");
        })
        .catch(() => {
            if (document.body.contains(reportRoot)) {
                document.body.removeChild(reportRoot);
            }
            toastLikeAlert("PDF generation failed. Try Preview instead.");
        });
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
