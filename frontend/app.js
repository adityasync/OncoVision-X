const uploadForm = document.getElementById("uploadForm");
const uploadInput = document.getElementById("ctScan");
const uploadZone = document.getElementById("uploadZone");
const selectedFile = document.getElementById("selectedFile");
const selectedFileName = document.getElementById("selectedFileName");
const selectedFileMeta = document.getElementById("selectedFileMeta");
const analyzeButton = document.getElementById("analyzeButton");
const loadingCard = document.getElementById("loadingCard");
const loadingDetail = document.getElementById("loadingDetail");
const loadingSteps = Array.from(document.querySelectorAll(".loading-steps span"));
const errorBanner = document.getElementById("errorBanner");
const resultsSection = document.getElementById("results");
const riskBanner = document.getElementById("riskBanner");
const riskIcon = document.getElementById("riskIcon");
const riskLabel = document.getElementById("riskLabel");
const riskSummary = document.getElementById("riskSummary");
const noduleCount = document.getElementById("noduleCount");
const analysisStatus = document.getElementById("analysisStatus");
const recommendationText = document.getElementById("recommendationText");
const timingGrid = document.getElementById("timingGrid");
const scanImage = document.getElementById("scanImage");
const scanPlaceholder = document.getElementById("scanPlaceholder");
const nodulesList = document.getElementById("nodulesList");
const emptyState = document.getElementById("emptyState");
const sampleButton = document.getElementById("sampleButton");
const exportButton = document.getElementById("exportButton");
const noduleCardTemplate = document.getElementById("noduleCardTemplate");

const loadingMessages = [
    "Processing volumetric slices",
    "Detecting suspicious nodules",
    "Calculating patient-level risk",
];

let loadingInterval = null;

uploadInput.addEventListener("change", () => {
    syncSelectedFiles(uploadInput.files);
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
    const files = event.dataTransfer?.files;
    if (!files?.length) {
        return;
    }

    uploadInput.files = files;
    syncSelectedFiles(files);
});

uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (!uploadInput.files.length) {
        showError("Select at least one scan file before starting analysis.");
        return;
    }

    hideError();
    setLoading(true);

    const formData = new FormData();
    Array.from(uploadInput.files).forEach((file) => {
        formData.append("ct_scan", file);
    });

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            body: formData,
        });

        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.error || "Analysis failed.");
        }

        renderResults(payload);
    } catch (error) {
        showError(error.message || "Analysis could not be completed.");
    } finally {
        setLoading(false);
    }
});

sampleButton.addEventListener("click", () => {
    renderResults({
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
    });
    document.getElementById("results").scrollIntoView({ behavior: "smooth", block: "start" });
});

exportButton.addEventListener("click", () => {
    if (!scanImage.src) {
        return;
    }

    const link = document.createElement("a");
    link.href = scanImage.src;
    link.download = "oncovision-ct-visualization.png";
    link.click();
});

function syncSelectedFiles(files) {
    if (!files?.length) {
        selectedFile.hidden = true;
        return;
    }

    const names = Array.from(files).map((file) => file.name);
    const size = Array.from(files).reduce((sum, file) => sum + file.size, 0);

    selectedFile.hidden = false;
    selectedFileName.textContent = names.length === 1 ? names[0] : `${names.length} files selected`;
    selectedFileMeta.textContent = `${formatBytes(size)} • Ready to analyze`;
}

function setLoading(isLoading) {
    analyzeButton.disabled = isLoading;
    analyzeButton.textContent = isLoading ? "Analyzing..." : "Analyze Scan";
    loadingCard.hidden = !isLoading;

    if (isLoading) {
        let index = 0;
        loadingDetail.textContent = loadingMessages[index];
        loadingSteps.forEach((step, stepIndex) => {
            step.classList.toggle("is-active", stepIndex === index);
        });

        loadingInterval = window.setInterval(() => {
            index = (index + 1) % loadingMessages.length;
            loadingDetail.textContent = loadingMessages[index];
            loadingSteps.forEach((step, stepIndex) => {
                step.classList.toggle("is-active", stepIndex === index);
            });
        }, 1400);
    } else if (loadingInterval !== null) {
        window.clearInterval(loadingInterval);
        loadingInterval = null;
    }
}

function renderResults(data) {
    const analysis = data.analysis || {};
    const overallRisk = (analysis.overall_risk || "LOW").toUpperCase();
    const nodules = analysis.nodules || [];
    const riskScore = Number(analysis.risk_score || 0).toFixed(1);

    resultsSection.hidden = false;
    resultsSection.classList.add("reveal");

    riskBanner.classList.remove("risk-banner--low", "risk-banner--medium", "risk-banner--high");
    riskBanner.classList.add(`risk-banner--${overallRisk.toLowerCase()}`);
    riskIcon.textContent = overallRisk.charAt(0);
    riskLabel.textContent = `Patient Risk: ${overallRisk}`;
    riskSummary.textContent = `Maximum malignancy: ${riskScore}%`;
    noduleCount.textContent = String(analysis.num_nodules_detected || 0);
    analysisStatus.textContent = data.status || "Analysis complete";
    recommendationText.textContent = data.next_steps || "Consult physician for evaluation.";

    renderTiming(data.timing || {});
    renderScan(data.visualization);
    renderNodules(nodules);

    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderTiming(timing) {
    timingGrid.innerHTML = "";

    const labels = [
        ["Preprocess", timing.preprocess_sec],
        ["Detection", timing.detect_sec],
        ["Total", timing.total_sec],
    ];

    labels.forEach(([label, value]) => {
        if (value == null) {
            return;
        }

        const item = document.createElement("div");
        item.innerHTML = `<span>${label}</span><strong>${Number(value).toFixed(1)}s</strong>`;
        timingGrid.appendChild(item);
    });
}

function renderScan(visualization) {
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

    nodules.forEach((nodule, index) => {
        const fragment = noduleCardTemplate.content.cloneNode(true);
        const root = fragment.querySelector(".nodule-card");
        const riskLevel = (nodule.risk_level || "LOW").toUpperCase();

        fragment.querySelector("h4").textContent = `Nodule #${nodule.nodule_id || index + 1}`;
        fragment.querySelector(".nodule-location").textContent = `Location: ${nodule.location}`;

        const pill = fragment.querySelector(".risk-pill");
        pill.textContent = riskLevel;
        pill.classList.add(`risk-pill--${riskLevel.toLowerCase()}`);

        const detectionValue = Number(nodule.detection_confidence || 0);
        const malignancyValue = Number(nodule.malignancy_probability || 0);

        fragment.querySelector(".detection-value").textContent = `${detectionValue.toFixed(1)}%`;
        fragment.querySelector(".malignancy-value").textContent = `${malignancyValue.toFixed(1)}%`;
        fragment.querySelector(".nodule-recommendation").textContent =
            nodule.recommendation || "Consult physician.";

        const detectionBar = fragment.querySelector(".detection-bar");
        const malignancyBar = fragment.querySelector(".malignancy-bar");
        detectionBar.style.background = progressGradient(detectionValue);
        malignancyBar.style.background = progressGradient(malignancyValue);

        nodulesList.appendChild(fragment);

        requestAnimationFrame(() => {
            detectionBar.style.width = `${detectionValue}%`;
            malignancyBar.style.width = `${malignancyValue}%`;
        });

        root?.style.setProperty("animation-delay", `${index * 100}ms`);
    });
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

function showError(message) {
    errorBanner.hidden = false;
    errorBanner.textContent = message;
}

function hideError() {
    errorBanner.hidden = true;
    errorBanner.textContent = "";
}

function formatBytes(bytes) {
    if (bytes === 0) {
        return "0 B";
    }

    const units = ["B", "KB", "MB", "GB"];
    const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    const value = bytes / 1024 ** index;
    return `${value.toFixed(value >= 100 || index === 0 ? 0 : 1)} ${units[index]}`;
}
