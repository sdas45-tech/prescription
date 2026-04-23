/**
 * script.js - Prescription Classifier Frontend Logic
 * Updated to align with train.py v2 (accuracy report support)
 */

// If running locally, connect to your local Python server. 
// If deployed to Vercel, connect to your hosted backend (e.g., Render, Railway).
const isLocalhost = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
const API_BASE = isLocalhost 
    ? "http://127.0.0.1:8000" 
    : "https://your-backend-url.onrender.com"; // <-- Change this to your deployed backend URL!

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

// ================================================================
// 6. Doctors App — Tab Navigation & Doctor Finder
// ================================================================

const SPECIALTIES_ICONS = {
    "General Physician":  "🩺",
    "Cardiologist":       "❤️",
    "Dermatologist":      "🔬",
    "Pediatrician":       "👶",
    "Orthopedist":        "🦴",
    "Neurologist":        "🧠",
    "Gynecologist":       "🩻",
    "Oncologist":         "🎗️",
    "ENT Specialist":     "👂"
};

const RATINGS = [4.2, 4.4, 4.5, 4.7, 4.8, 4.9, 5.0];

let allDoctors = [];
let doctorsLoaded = false;

// Tab switching
document.getElementById("tabScanner").addEventListener("click", () => switchTab("scanner"));
document.getElementById("tabDoctors").addEventListener("click", () => {
    switchTab("doctors");
    if (!doctorsLoaded) loadDoctors();
});

function switchTab(tab) {
    const scannerMain = document.querySelector(".main-content:not(#doctorsTab)");
    const doctorsMain = document.getElementById("doctorsTab");
    const tabScanner  = document.getElementById("tabScanner");
    const tabDoctors  = document.getElementById("tabDoctors");

    if (tab === "scanner") {
        scannerMain.style.display = "flex";
        doctorsMain.style.display = "none";
        tabScanner.classList.add("active");
        tabDoctors.classList.remove("active");
    } else {
        scannerMain.style.display = "none";
        doctorsMain.style.display = "block";
        tabDoctors.classList.add("active");
        tabScanner.classList.remove("active");
    }
}

// Load doctors from backend
function loadDoctors() {
    if (!navigator.geolocation) {
        renderDoctorError("Geolocation not supported by your browser.");
        return;
    }

    navigator.geolocation.getCurrentPosition(async (pos) => {
        const lat = pos.coords.latitude;
        const lon = pos.coords.longitude;
        const radius = 10000;

        try {
            const response = await fetch(`${API_BASE}/api/hospitals?lat=${lat}&lon=${lon}&radius=${radius}`);
            if (!response.ok) throw new Error(`Server returned ${response.status}`);
            const data = await response.json();

            // Flatten all doctors from all hospitals
            allDoctors = [];
            const hospitals = data.hospitals || [];

            hospitals.forEach(h => {
                (h.roster || []).forEach(doc => {
                    const rating = RATINGS[Math.floor(Math.random() * RATINGS.length)];
                    allDoctors.push({
                        name:      doc.name,
                        specialty: doc.specialty,
                        status:    doc.status,
                        hospital:  h.name,
                        phone:     h.phone,
                        opd:       h.opd_time,
                        lat:       h.lat,
                        lon:       h.lon,
                        rating:    rating,
                        icon:      SPECIALTIES_ICONS[doc.specialty] || "👨‍⚕️",
                        mapLink:   `https://www.google.com/maps/dir/?api=1&destination=${h.lat},${h.lon}`
                    });
                });
            });

            doctorsLoaded = true;
            updateStats(hospitals.length);
            renderDoctors(allDoctors);
            setupDoctorFilters();

        } catch (err) {
            renderDoctorError(`Failed to load doctors: ${err.message}`);
        }
    }, () => {
        renderDoctorError("Location access denied. Please enable location access in your browser settings.");
    });
}

function updateStats(hospitalCount) {
    const total     = allDoctors.length;
    const available = allDoctors.filter(d => d.status === "Available").length;

    document.getElementById("statTotal").textContent     = total;
    document.getElementById("statAvailable").textContent = available;
    document.getElementById("statHospitals").textContent = hospitalCount;
}

function renderDoctors(doctors) {
    const grid = document.getElementById("doctorGrid");

    if (doctors.length === 0) {
        grid.innerHTML = `
            <div class="no-doctors">
                <span class="no-doctors-icon">🔍</span>
                <p>No doctors match your search. Try changing filters.</p>
            </div>`;
        return;
    }

    grid.innerHTML = doctors.map((doc, i) => {
        const stars = "⭐".repeat(Math.floor(doc.rating));
        return `
            <div class="doctor-card" style="animation-delay: ${i * 0.05}s;" onclick="openDoctorModal(${i})">
                <div class="doctor-card-top">
                    <div class="doctor-avatar">${doc.icon}</div>
                    <div class="doctor-info">
                        <div class="doctor-name">${doc.name}</div>
                        <div class="doctor-specialty">${doc.specialty}</div>
                        <div class="doctor-hospital">🏥 ${doc.hospital}</div>
                    </div>
                    <div class="facility-status-badge" style="background:#22c55e22; color:#22c55e; border:1px solid #22c55e44; white-space:nowrap;">
                        ✅ Available
                    </div>
                </div>

                <div class="doctor-status-row">
                    <div class="doctor-rating">⭐ ${doc.rating.toFixed(1)}</div>
                    <div class="doctor-opd">🕐 ${doc.opd}</div>
                </div>

                <div class="doctor-actions">
                    <a href="tel:${doc.phone}" class="btn-action btn-action-call" onclick="event.stopPropagation()">📞 Call</a>
                    <a href="${doc.mapLink}" target="_blank" class="btn-action btn-action-map" onclick="event.stopPropagation()">📍 Map</a>
                    <button class="btn-action" onclick="event.stopPropagation(); openBookModal(${i})" style="flex:1; background:rgba(139,92,246,0.12); color:var(--accent-purple-light); border:1px solid rgba(139,92,246,0.25); border-radius:8px; font-weight:600; font-size:0.85rem; cursor:pointer; padding:10px 0; transition:all 0.2s ease;" onmouseover="this.style.background='var(--accent-purple)';this.style.color='white';" onmouseout="this.style.background='rgba(139,92,246,0.12)';this.style.color='var(--accent-purple-light)';">📅 Book</button>
                </div>
            </div>`;
    }).join("");

    // store filtered array for modal access
    window._renderedDoctors = doctors;
}

function openBookModal(idx) {
    const doc = (window._renderedDoctors || allDoctors)[idx];
    if (!doc) return;
    const modal = document.getElementById("bookModal");
    document.getElementById("modalContent").innerHTML = `
        <h3 style="font-size:1.2rem; font-weight:800; color:var(--text-primary); margin-bottom:4px;">Book Appointment</h3>
        <p style="color:var(--text-muted); font-size:0.85rem; margin-bottom:24px;">Fill in your details to schedule with <strong style="color:var(--accent-purple-light);">${doc.name}</strong></p>

        <div style="display:flex; align-items:center; gap:14px; padding:14px; background:rgba(0,0,0,0.2); border-radius:12px; margin-bottom:24px; border:1px solid var(--border-color);">
            <div style="width:48px;height:48px;border-radius:50%;background:var(--gradient-primary);display:flex;align-items:center;justify-content:center;font-size:1.4rem;">${doc.icon}</div>
            <div>
                <div style="font-weight:700; color:var(--text-primary);">${doc.name}</div>
                <div style="font-size:0.78rem; color:var(--accent-blue); text-transform:uppercase; letter-spacing:0.05em;">${doc.specialty}</div>
                <div style="font-size:0.78rem; color:var(--text-muted);">🏥 ${doc.hospital}</div>
            </div>
        </div>

        <div style="display:flex; flex-direction:column; gap:14px; margin-bottom:24px;">
            <input type="text" placeholder="Your Full Name" style="padding:12px 16px; background:rgba(0,0,0,0.25); border:1px solid var(--border-color); border-radius:10px; color:var(--text-primary); font-family:var(--font); font-size:0.95rem; outline:none; width:100%;" />
            <input type="tel" placeholder="Your Phone Number" style="padding:12px 16px; background:rgba(0,0,0,0.25); border:1px solid var(--border-color); border-radius:10px; color:var(--text-primary); font-family:var(--font); font-size:0.95rem; outline:none; width:100%;" />
            <input type="date" style="padding:12px 16px; background:rgba(0,0,0,0.25); border:1px solid var(--border-color); border-radius:10px; color:var(--text-primary); font-family:var(--font); font-size:0.95rem; outline:none; width:100%; color-scheme: dark;" />
            <select style="padding:12px 16px; background:rgba(0,0,0,0.25); border:1px solid var(--border-color); border-radius:10px; color:var(--text-primary); font-family:var(--font); font-size:0.88rem; outline:none; width:100%;">
                <option>Morning (9 AM - 12 PM)</option>
                <option>Afternoon (12 PM - 4 PM)</option>
                <option>Evening (4 PM - 8 PM)</option>
            </select>
        </div>

        <button onclick="confirmBooking('${doc.name}')" style="width:100%; padding:14px; background:var(--gradient-primary); border:none; border-radius:12px; color:white; font-family:var(--font); font-size:1rem; font-weight:700; cursor:pointer; transition:all 0.2s ease;" onmouseover="this.style.transform='translateY(-2px)';this.style.boxShadow='0 8px 24px rgba(139,92,246,0.4)';" onmouseout="this.style.transform='';this.style.boxShadow='';">
            ✅ Confirm Appointment
        </button>`;
    modal.style.display = "flex";
}

function openDoctorModal(idx) {
    openBookModal(idx);
}

function confirmBooking(docName) {
    document.getElementById("modalContent").innerHTML = `
        <div style="text-align:center; padding: 20px 0;">
            <div style="font-size:4rem; margin-bottom:16px;">✅</div>
            <h3 style="font-size:1.3rem; font-weight:800; color:var(--accent-green); margin-bottom:8px;">Appointment Requested!</h3>
            <p style="color:var(--text-muted); line-height:1.6;">Your appointment request with <strong style="color:var(--text-primary);">${docName}</strong> has been submitted. You will receive a confirmation shortly.</p>
            <button onclick="document.getElementById('bookModal').style.display='none'" style="margin-top:24px; padding:12px 32px; background:var(--gradient-primary); border:none; border-radius:12px; color:white; font-family:var(--font); font-size:0.95rem; font-weight:700; cursor:pointer;">Close</button>
        </div>`;
}

document.getElementById("modalClose").addEventListener("click", () => {
    document.getElementById("bookModal").style.display = "none";
});
document.getElementById("bookModal").addEventListener("click", (e) => {
    if (e.target.id === "bookModal") document.getElementById("bookModal").style.display = "none";
});

function renderDoctorError(msg) {
    document.getElementById("doctorGrid").innerHTML = `
        <div class="no-doctors">
            <span class="no-doctors-icon">⚠️</span>
            <p>${msg}</p>
        </div>`;
}

function setupDoctorFilters() {
    const search   = document.getElementById("doctorSearch");
    const specFilt = document.getElementById("specialtyFilter");
    const availFilt = document.getElementById("availabilityFilter");

    function applyFilters() {
        const q     = search.value.toLowerCase();
        const spec  = specFilt.value;
        const avail = availFilt.value;

        const filtered = allDoctors.filter(doc => {
            const matchQ    = !q || doc.name.toLowerCase().includes(q) || doc.specialty.toLowerCase().includes(q) || doc.hospital.toLowerCase().includes(q);
            const matchSpec  = !spec  || doc.specialty === spec;
            const matchAvail = !avail || doc.status === avail;
            return matchQ && matchSpec && matchAvail;
        });

        renderDoctors(filtered);
    }

    search.addEventListener("input", applyFilters);
    specFilt.addEventListener("change", applyFilters);
    availFilt.addEventListener("change", applyFilters);
}

