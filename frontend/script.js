/**
 * script.js - Prescription Classifier Frontend Logic
 * Updated to align with train.py v2 (accuracy report support)
 */

const API_BASE = window.location.origin;

// -- DOM Elements --
const dropZone        = document.getElementById("dropZone");
const dropZoneContent = document.getElementById("dropZoneContent");
const previewContainer = document.getElementById("previewContainer");
const previewImage    = document.getElementById("previewImage");
const fileInput       = document.getElementById("fileInput");
const btnPredict      = document.getElementById("btnPredict");
const btnLoader       = document.getElementById("btnLoader");
const btnRemove       = document.getElementById("btnRemove");
const btnNew          = document.getElementById("btnNew");
const btnClearHistory = document.getElementById("btnClearHistory");

const uploadSection   = document.getElementById("uploadSection");
const resultsSection  = document.getElementById("resultsSection");
const historySection  = document.getElementById("historySection");
const historyList     = document.getElementById("historyList");

const modelStatus     = document.getElementById("modelStatus");
const statusDot       = modelStatus.querySelector(".status-dot");
const statusText      = modelStatus.querySelector(".status-text");

// Result elements
const resultBadge     = document.getElementById("resultBadge");
const badgeIcon       = document.getElementById("badgeIcon");
const badgeLabel      = document.getElementById("badgeLabel");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar   = document.getElementById("confidenceBar");
const detailClass     = document.getElementById("detailClass");
const detailScore     = document.getElementById("detailScore");
const detailFile      = document.getElementById("detailFile");
const detailSize      = document.getElementById("detailSize");


let selectedFile = null;
let predictionHistory = JSON.parse(localStorage.getItem("pred_history") || "[]");
let map = null;
let markerGroup = null;

// ================================================================
// 1. Initialization
// ================================================================
document.addEventListener("DOMContentLoaded", () => {
    checkModelStatus();
    renderHistory();
    
    // Add event listener for map radius changes
    const radiusSelect = document.getElementById("radiusSelect");
    if (radiusSelect) {
        radiusSelect.addEventListener("change", () => {
            if (document.getElementById("mapSection").style.display === "block") {
                initMapAndFindHospitals();
            }
        });
    }
});

async function checkModelStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();

        if (data.model_loaded) {
            statusDot.classList.add("active");
            statusText.textContent = "Model Ready";
        } else {
            statusDot.classList.add("error");
            statusText.textContent = "Model Not Loaded";
        }
    } catch {
        statusDot.classList.add("error");
        statusText.textContent = "Server Offline";
    }
}


// ================================================================
// 2. File Upload Handling
// ================================================================
dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

// Drag & Drop
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith("image/")) {
        alert("Please upload an image file (JPG, PNG, WebP, BMP).");
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        dropZoneContent.style.display = "none";
        previewContainer.style.display = "flex";
        btnPredict.disabled = false;
    };
    reader.readAsDataURL(file);
}

btnRemove.addEventListener("click", (e) => {
    e.stopPropagation();
    resetUpload();
});

function resetUpload() {
    selectedFile = null;
    fileInput.value = "";
    previewImage.src = "";
    dropZoneContent.style.display = "flex";
    previewContainer.style.display = "none";
    btnPredict.disabled = true;
}

// ================================================================
// 3. Prediction
// ================================================================
btnPredict.addEventListener("click", analyzePrediction);

async function analyzePrediction() {
    if (!selectedFile) return;

    // Show loading state
    btnPredict.disabled = true;
    document.querySelector(".btn-text").style.display = "none";
    document.querySelector(".btn-icon").style.display = "none";
    btnLoader.style.display = "flex";

    try {
        const formData = new FormData();
        formData.append("file", selectedFile);

        const res = await fetch(`${API_BASE}/api/predict`, {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Prediction failed");
        }

        const data = await res.json();
        showResults(data);

        // Save to history
        addToHistory({
            filename: data.file_info.filename,
            label: data.prediction.label,
            confidence: data.prediction.confidence,
            timestamp: new Date().toISOString(),
        });

    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        // Reset button
        document.querySelector(".btn-text").style.display = "inline";
        document.querySelector(".btn-icon").style.display = "inline";
        btnLoader.style.display = "none";
        btnPredict.disabled = false;
    }
}

// ================================================================
// 4. Show Results
// ================================================================
function showResults(data) {
    const { prediction, file_info } = data;
    const isPrescription = prediction.is_prescription;

    // Hide upload, show results
    uploadSection.style.display = "none";
    resultsSection.style.display = "block";

    // Badge
    resultBadge.className = `result-badge ${isPrescription ? "prescription" : "not-prescription"}`;
    badgeIcon.textContent = isPrescription ? "\uD83D\uDCCB" : "\uD83D\uDEAB";
    badgeLabel.textContent = prediction.label.replace("_", " ").toUpperCase();

    // Confidence
    const confPercent = (prediction.confidence * 100).toFixed(1);
    confidenceValue.textContent = `${confPercent}%`;

    // Animate confidence bar
    confidenceBar.style.width = "0%";
    confidenceBar.style.background = isPrescription
        ? "linear-gradient(135deg, #22c55e 0%, #16a34a 100%)"
        : "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)";
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            confidenceBar.style.width = `${confPercent}%`;
        });
    });

    // Details
    detailClass.textContent = prediction.label.replace("_", " ");
    detailScore.textContent = prediction.raw_score.toFixed(4);
    detailFile.textContent = file_info.filename;
    detailSize.textContent = `${file_info.image_size[0]} x ${file_info.image_size[1]} px`;

    // AI Layman Translation
    const aiAnalysisBox = document.getElementById("aiAnalysisBox");
    const aiAnalysisText = document.getElementById("aiAnalysisText");
    
    if (isPrescription && prediction.layman_explanation) {
        aiAnalysisBox.style.display = "block";
        
        // Simple markdown parsing for Gemini response
        let formattedText = prediction.layman_explanation
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n\*/g, '<br>• ')
            .replace(/\n\-/g, '<br>• ')
            .replace(/\n/g, '<br>');
            
        aiAnalysisText.innerHTML = formattedText;
        
        // Show Map Section and find hospitals
        document.getElementById("mapSection").style.display = "block";
        initMapAndFindHospitals();
    } else {
        aiAnalysisBox.style.display = "none";
        document.getElementById("mapSection").style.display = "none";
    }
}

function initMapAndFindHospitals() {
    const mapStatus = document.getElementById("mapStatus");
    const radiusSelect = document.getElementById("radiusSelect");
    const radius = radiusSelect ? parseInt(radiusSelect.value) : 5000;
    
    if (!navigator.geolocation) {
        mapStatus.textContent = "Geolocation is not supported by your browser.";
        return;
    }

    navigator.geolocation.getCurrentPosition(
        async (position) => {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;
            
            mapStatus.textContent = `Location found. Fetching medical facilities from API...`;
            
            if (!map) {
                map = L.map('map').setView([lat, lon], 13);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '&copy; OpenStreetMap contributors'
                }).addTo(map);
                markerGroup = L.layerGroup().addTo(map);
            } else {
                map.setView([lat, lon], 13);
                markerGroup.clearLayers();
            }
            
            // Add user marker
            L.marker([lat, lon]).addTo(markerGroup)
                .bindPopup("<b>You are here</b>").openPopup();

            try {
                // Fetch enriched data from our own backend API
                const response = await fetch(`${API_BASE}/api/hospitals?lat=${lat}&lon=${lon}&radius=${radius}`);
                
                if (!response.ok) {
                    const errText = await response.text();
                    throw new Error(`Server Error: ${response.status}`);
                }

                const data = await response.json();
                
                if (data.error) throw new Error(data.error);
                
                const hospitals = data.hospitals || [];
                if (hospitals.length === 0) {
                    mapStatus.textContent = `No medical facilities found within ${radius/1000}km.`;
                    document.getElementById("facilityList").style.display = "none";
                    return;
                }
                
                mapStatus.textContent = `Found ${hospitals.length} medical facilities nearby.`;
                
                const facilityGrid = document.getElementById("facilityGrid");
                if (facilityGrid) facilityGrid.innerHTML = "";
                document.getElementById("facilityList").style.display = "block";
                
                hospitals.forEach(h => {
                    let doctorsHtml = h.roster.map(doc => {
                        const color = doc.status === "Available" ? "#22c55e" : "#eab308";
                        return `
                            <li style="margin-bottom:6px; display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #eee; padding-bottom:4px;">
                                <div>
                                    <div style="font-weight:600; color:#111;">${doc.name}</div>
                                    <div style="font-size:11px; color:#666;">${doc.specialty}</div>
                                </div>
                                <div style="font-size:10px; padding:2px 6px; border-radius:10px; background:${color}22; color:${color}; font-weight:700; border:1px solid ${color}44;">
                                    ${doc.status}
                                </div>
                            </li>`;
                    }).join("");

                    const mapLink = `https://www.google.com/maps/dir/?api=1&destination=${h.lat},${h.lon}`;

                    const popupContent = `
                        <div style="font-size:13px; min-width:240px; max-height:300px; overflow-y:auto; font-family:'Inter', sans-serif; padding:4px;">
                            <h4 style="margin:0 0 4px 0; color:#1a56db; font-size:15px; font-weight:700;">${h.name}</h4>
                            <div style="font-size:11px; color:#666; margin-bottom:10px; text-transform:uppercase; letter-spacing:0.5px;">${h.type} &bull; Open Now</div>
                            
                            <div style="background:#f8fafc; padding:8px; border-radius:6px; margin-bottom:12px; border:1px solid #e2e8f0;">
                                <div style="margin-bottom:4px;"><b>OPD:</b> ${h.opd_time}</div>
                                <div style="margin-bottom:0px;"><b>Contact:</b> ${h.phone}</div>
                            </div>
                            
                            <div style="margin-bottom:12px;">
                                <b style="display:block; margin-bottom:8px; color:#334155; font-size:12px; text-transform:uppercase;">Specialist Roster</b>
                                <ul style="margin:0; padding:0; list-style:none;">
                                    ${doctorsHtml}
                                </ul>
                            </div>
                            
                            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin-top:8px;">
                                <a href="tel:${h.phone}" style="background:#10b981; color:white; text-decoration:none; padding:8px 0; border-radius:6px; font-weight:600; text-align:center; font-size:12px;">
                                    📞 Call Now
                                </a>
                                <a href="${mapLink}" target="_blank" style="background:#2563eb; color:white; text-decoration:none; padding:8px 0; border-radius:6px; font-weight:600; text-align:center; font-size:12px;">
                                    📍 Directions
                                </a>
                            </div>
                        </div>
                    `;
                    
                    L.marker([h.lat, h.lon]).addTo(markerGroup)
                        .bindPopup(popupContent);
                        
                    // Add to grid
                    if (facilityGrid) {
                        const gridCard = document.createElement("div");
                        gridCard.className = "facility-card";
                        
                        let cardRosterHtml = h.roster.map(doc => {
                            const badgeColor = doc.status === "Available" ? "#22c55e" : "#f59e0b";
                            return `
                                <div class="facility-roster-item">
                                    <div>
                                        <div class="facility-roster-name">${doc.name}</div>
                                        <div class="facility-roster-spec">${doc.specialty}</div>
                                    </div>
                                    <div class="facility-status-badge" style="background: ${badgeColor}22; color: ${badgeColor}; border: 1px solid ${badgeColor}44;">
                                        ${doc.status}
                                    </div>
                                </div>
                            `;
                        }).join('');

                        gridCard.innerHTML = `
                            <h4 class="facility-card-title">${h.name}</h4>
                            <div class="facility-card-meta">
                                <span>${h.type}</span> &bull; <span>OPD: ${h.opd_time}</span>
                            </div>
                            <div class="facility-roster">
                                ${cardRosterHtml}
                            </div>
                            <div class="facility-actions">
                                <a href="tel:${h.phone}" class="btn-action btn-action-call">📞 Call Now</a>
                                <a href="${mapLink}" target="_blank" class="btn-action btn-action-map">📍 Directions</a>
                            </div>
                        `;
                        facilityGrid.appendChild(gridCard);
                    }
                });

            } catch (err) {
                mapStatus.textContent = `Error: ${err.message}`;
                console.error(err);
            }
        },
        (error) => {
            mapStatus.textContent = "Unable to retrieve your location. Please allow location access in your browser settings.";
        }
    );
}

// -- New Analysis --
btnNew.addEventListener("click", () => {
    resultsSection.style.display = "none";
    document.getElementById("mapSection").style.display = "none";
    uploadSection.style.display = "block";
    resetUpload();
});

// ================================================================
// 5. History
// ================================================================
function addToHistory(item) {
    predictionHistory.unshift(item);
    if (predictionHistory.length > 20) predictionHistory.pop();
    localStorage.setItem("pred_history", JSON.stringify(predictionHistory));
    renderHistory();
}

function renderHistory() {
    if (predictionHistory.length === 0) {
        historySection.style.display = "none";
        return;
    }

    historySection.style.display = "block";
    historyList.innerHTML = predictionHistory.map((item) => {
        const isPres = item.label === "prescription";
        return `
            <div class="history-item">
                <span class="history-icon">${isPres ? "\uD83D\uDCCB" : "\uD83D\uDEAB"}</span>
                <div class="history-info">
                    <div class="history-name">${item.filename}</div>
                    <div class="history-label">${item.label.replace("_", " ")} - ${new Date(item.timestamp).toLocaleTimeString()}</div>
                </div>
                <span class="history-conf">${(item.confidence * 100).toFixed(1)}%</span>
            </div>
        `;
    }).join("");
}

btnClearHistory.addEventListener("click", () => {
    predictionHistory = [];
    localStorage.removeItem("pred_history");
    renderHistory();
});
