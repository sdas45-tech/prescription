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

// Specialty icons shared across map popups and doctor tab
const SPECIALTIES_ICONS_MAP = {
    "General Physician": "🩺",
    "Cardiologist":      "❤️",
    "Dermatologist":     "🔬",
    "Pediatrician":      "👶",
    "Orthopedist":       "🦴",
    "Neurologist":       "🧠",
    "Gynecologist":      "🩻",
    "Oncologist":        "🎗️",
    "ENT Specialist":    "👂"
};


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

        // Store raw text for language switching and voice output
        const rawText = prediction.layman_explanation;
        window._rawAiAnalysis = rawText;

        // Reset language selector to English on each new scan
        const langSel = document.getElementById("languageSelect");
        if (langSel) langSel.value = "English";

        // Reset interaction/allergy panels
        document.getElementById("interactionPanel").style.display = "none";
        document.getElementById("allergyPanel").style.display     = "none";

        // Simple markdown parsing for Gemini response
        let formattedText = rawText
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n\*/g, '<br>• ')
            .replace(/\n\-/g, '<br>• ')
            .replace(/\n/g, '<br>');

        aiAnalysisText.innerHTML = formattedText;

        // ----- Show OCR Panel -----
        const ocr = data.ocr || {};
        renderOcrPanel(ocr);

        // Use OCR medicines (accurate) if available, otherwise fall back to AI text heuristic
        const medicinesForCheck = (ocr.medicines && ocr.medicines.length > 0)
            ? ocr.medicines
            : null; // runSafetyChecks will extract from AI text as fallback

        // Show Map Section and find hospitals
        document.getElementById("mapSection").style.display = "block";
        initMapAndFindHospitals();

        // Run drug interaction + allergy safety checks
        runSafetyChecks(rawText, medicinesForCheck);

    } else {
        aiAnalysisBox.style.display = "none";
        document.getElementById("mapSection").style.display       = "none";
        document.getElementById("interactionPanel").style.display = "none";
        document.getElementById("allergyPanel").style.display     = "none";
        const ocrPanel = document.getElementById("ocrPanel");
        if (ocrPanel) ocrPanel.style.display = "none";
        window._rawAiAnalysis = null;
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
                    const specs = h.available_specialties || [];
                    const mapLink = `https://www.google.com/maps/dir/?api=1&destination=${h.lat},${h.lon}`;

                    const specListHtml = specs.map(spec => `
                        <li style="padding:5px 0; border-bottom:1px solid #e2e8f0; font-size:12px; color:#334155;">
                            ${SPECIALTIES_ICONS_MAP[spec] || '💊'} ${spec}
                        </li>`).join("");

                    const popupContent = `
                        <div style="font-size:13px; min-width:240px; max-height:300px; overflow-y:auto; font-family:'Inter', sans-serif; padding:4px;">
                            <h4 style="margin:0 0 4px 0; color:#1a56db; font-size:15px; font-weight:700;">${h.name}</h4>
                            <div style="font-size:11px; color:#666; margin-bottom:10px; text-transform:uppercase;">${h.type} &bull; Open Now</div>
                            <div style="background:#f8fafc; padding:8px; border-radius:6px; margin-bottom:10px; border:1px solid #e2e8f0;">
                                <div><b>OPD:</b> ${h.opd_time}</div>
                                <div><b>Contact:</b> ${h.phone}</div>
                            </div>
                            <div style="margin-bottom:10px;">
                                <b style="display:block; margin-bottom:6px; color:#334155; font-size:11px; text-transform:uppercase;">Available Specialties (${specs.length})</b>
                                <ul style="margin:0; padding:0; list-style:none;">${specListHtml}</ul>
                            </div>
                            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                                <a href="tel:${h.phone}" style="background:#10b981; color:white; text-decoration:none; padding:8px 0; border-radius:6px; font-weight:600; text-align:center; font-size:12px;">📞 Call</a>
                                <a href="${mapLink}" target="_blank" style="background:#2563eb; color:white; text-decoration:none; padding:8px 0; border-radius:6px; font-weight:600; text-align:center; font-size:12px;">📍 Directions</a>
                            </div>
                        </div>`;
                    
                    L.marker([h.lat, h.lon]).addTo(markerGroup).bindPopup(popupContent);
                        
                    // Add to facility grid below map
                    if (facilityGrid) {
                        const gridCard = document.createElement("div");
                        gridCard.className = "facility-card";

                        const badgeHtml = specs.map(spec => `
                            <span style="display:inline-flex; align-items:center; gap:4px; background:rgba(139,92,246,0.12); color:var(--accent-purple-light); border:1px solid rgba(139,92,246,0.2); border-radius:20px; padding:4px 10px; font-size:0.72rem; font-weight:600;">
                                ${SPECIALTIES_ICONS_MAP[spec] || '💊'} ${spec}
                            </span>`).join("");

                        gridCard.innerHTML = `
                            <h4 class="facility-card-title">${h.name}</h4>
                            <div class="facility-card-meta"><span>${h.type.toUpperCase()}</span> &bull; <span>OPD: ${h.opd_time}</span></div>
                            <div style="margin-bottom:14px; display:flex; flex-wrap:wrap; gap:6px;">${badgeHtml}</div>
                            <div class="facility-actions">
                                <a href="tel:${h.phone}" class="btn-action btn-action-call">📞 Call Now</a>
                                <a href="${mapLink}" target="_blank" class="btn-action btn-action-map">📍 Directions</a>
                            </div>`;
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
// 6. New Features: Voice Output, Language, Allergies, Interactions
// ================================================================

// --- Allergy Manager (localStorage) ---
let userAllergies = JSON.parse(localStorage.getItem("pharma_allergies") || "[]");

function renderAllergyTags() {
    const wrap = document.getElementById("allergyTags");
    if (!wrap) return;
    if (userAllergies.length === 0) {
        wrap.innerHTML = `<span style="font-size:0.82rem; color:var(--text-muted);">No allergies saved yet.</span>`;
        return;
    }
    wrap.innerHTML = userAllergies.map((a, i) => `
        <span class="allergy-tag">
            🚨 ${a}
            <button onclick="removeAllergy(${i})" title="Remove">×</button>
        </span>`).join("");
}

function addAllergy() {
    const input = document.getElementById("allergyInput");
    const val   = input.value.trim();
    if (!val) return;
    if (!userAllergies.includes(val)) {
        userAllergies.push(val);
        localStorage.setItem("pharma_allergies", JSON.stringify(userAllergies));
    }
    input.value = "";
    renderAllergyTags();
}

function removeAllergy(idx) {
    userAllergies.splice(idx, 1);
    localStorage.setItem("pharma_allergies", JSON.stringify(userAllergies));
    renderAllergyTags();
}

// Allow Enter key to add allergy
document.getElementById("allergyInput")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") addAllergy();
});

renderAllergyTags(); // init on load

// --- Voice Output (Web Speech API) ---
let isSpeaking = false;
let speechChunks = [];

function speakAnalysis() {
    const btn  = document.getElementById("voiceBtn");
    // Get visible text and strip out markdown symbols or emojis to help the speech engine
    let text = document.getElementById("aiAnalysisText")?.innerText || "";
    text = text.replace(/[\*•🩺👨‍⚕️💊🚨]/g, "").trim();

    if (!text || text === "--") return;

    if (isSpeaking) {
        window.speechSynthesis.cancel();
        isSpeaking = false;
        btn.innerHTML = `🔊 <span class="feature-btn-label">Listen</span>`;
        btn.classList.remove("voice-btn-active");
        return;
    }

    const lang    = document.getElementById("languageSelect")?.value || "English";
    const langMap = {
        "Hindi": "hi-IN", "Bengali": "bn-IN", "Tamil": "ta-IN", "Telugu": "te-IN",
        "Marathi": "mr-IN", "Gujarati": "gu-IN", "Kannada": "kn-IN",
        "Malayalam": "ml-IN", "Punjabi": "pa-IN", "Urdu": "ur-PK",
        "Spanish": "es-ES", "French": "fr-FR", "German": "de-DE",
        "Arabic": "ar-SA", "Chinese": "zh-CN", "Japanese": "ja-JP",
        "English": "en-US"
    };

    window.speechSynthesis.cancel();

    // Chrome has a bug where speech synthesis stops after 15 seconds.
    // To fix this, we split the text into chunks and speak them sequentially.
    speechChunks = text.match(/[^.!?]+[.!?]+/g) || [text];
    let currentChunk = 0;
    isSpeaking = true;

    btn.innerHTML = `⏹ <span class="feature-btn-label">Stop</span>`;
    btn.classList.add("voice-btn-active");

    function speakNextChunk() {
        if (!isSpeaking || currentChunk >= speechChunks.length) {
            isSpeaking = false;
            btn.innerHTML = `🔊 <span class="feature-btn-label">Listen</span>`;
            btn.classList.remove("voice-btn-active");
            return;
        }

        const utterance = new SpeechSynthesisUtterance(speechChunks[currentChunk].trim());
        utterance.lang  = langMap[lang] || "en-US";
        utterance.rate  = 0.9;
        utterance.pitch = 1;

        utterance.onend = () => {
            currentChunk++;
            speakNextChunk();
        };

        utterance.onerror = (e) => {
            console.error("Speech error:", e);
            isSpeaking = false;
            btn.innerHTML = `🔊 <span class="feature-btn-label">Listen</span>`;
            btn.classList.remove("voice-btn-active");
        };

        window.speechSynthesis.speak(utterance);
    }

    speakNextChunk();
}

// --- Simple Markdown Formatter Helper ---
function formatMarkdown(text) {
    if (typeof marked !== "undefined") {
        return marked.parse(text);
    }
    // Fallback if marked fails to load
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n\*/g, '<br>• ')
        .replace(/\n\-/g, '<br>• ')
        .replace(/\n/g, '<br>');
}

// --- Multi-Language Translation ---
document.getElementById("languageSelect")?.addEventListener("change", async function () {
    const lang = this.value;
    const rawText = window._rawAiAnalysis;
    if (!rawText || rawText === "--") return;

    // Stop speaking if translation changes
    if (isSpeaking) speakAnalysis();

    const loader = document.getElementById("translateLoader");
    const box    = document.getElementById("aiAnalysisText");

    if (lang === "English") {
        box.innerHTML = formatMarkdown(rawText);
        return;
    }

    loader.style.display = "block";
    try {
        const resp = await fetch(`${API_BASE}/api/translate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: rawText, language: lang })
        });
        const data = await resp.json();
        if (data.translated) {
            box.innerHTML = formatMarkdown(data.translated);
        } else {
            box.innerHTML = `<span style="color:var(--accent-amber);">${data.warning || data.error || "Translation failed."}</span>`;
        }
    } catch (e) {
        box.innerHTML = `<span style="color:var(--accent-red);">Translation error: ${e.message}</span>`;
    } finally {
        loader.style.display = "none";
    }
});

// --- Extract medicine names from AI analysis text ---
function extractMedicineNames(text) {
    if (!text) return [];
    // Heuristic: match capitalized words near dosage patterns (mg, ml, tablet, capsule)
    const matches = text.match(/\b[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?\b(?=[\s\S]{0,60}?(?:mg|ml|tablet|capsule|dose|daily|twice|thrice|morning|evening))/gi) || [];
    // Deduplicate and filter short common words
    const stopWords = new Set(["The","This","Please","After","Before","You","Your","For","With","Take","Each","And","Has","Are","Not","Any","Its","How","When","That","They","Have","Been","Will","Can","Also","Such","More","All","From","Into","What","About","Some","Than","Then","Both","Only","Very","Most","Much","Even","Just","Does","Same","High","Low","Risk","Side","Drug","Help","Note","Sign","Tell","Food","Blood","Heart","Body","Time","Once","Day","Week","Month","Year","Days","Use","Used","May","Possible","Common","If"]);
    return [...new Set(matches.filter(m => m.length > 3 && !stopWords.has(m)))].slice(0, 12);
}

// --- OCR Panel Renderer ---
function renderOcrPanel(ocr) {
    let panel = document.getElementById("ocrPanel");
    if (!panel) return;

    if (!ocr.available) {
        panel.style.display = "none";
        return;
    }

    panel.style.display = "block";
    const qualityColors = { good: "badge-ok", poor: "badge-warning", unreadable: "badge-danger", error: "badge-danger" };
    const qualityClass  = qualityColors[ocr.quality] || "badge-warning";

    const medsHtml = (ocr.medicines || []).length > 0
        ? ocr.medicines.map(m => `<span class="allergy-tag" style="background:rgba(59,130,246,0.12);color:#3b82f6;border-color:rgba(59,130,246,0.3);">💊 ${m}</span>`).join("")
        : `<span style="font-size:0.8rem;color:var(--text-muted);">No medicines extracted</span>`;

    document.getElementById("ocrQualityBadge").textContent = ocr.quality?.toUpperCase() || "UNKNOWN";
    document.getElementById("ocrQualityBadge").className   = `feature-badge ${qualityClass}`;
    document.getElementById("ocrWordCount").textContent    = `${ocr.word_count || 0} words`;
    document.getElementById("ocrRawText").textContent      = ocr.text || "(no text extracted)";
    document.getElementById("ocrMedicines").innerHTML      = medsHtml;
}

// --- Run Interaction + Allergy checks after a prescription is analysed ---
async function runSafetyChecks(aiText, ocrMedicines = null) {
    // Prefer accurate OCR-extracted medicines; fall back to JS heuristic on AI text
    const medicines = ocrMedicines || extractMedicineNames(aiText);
    if (medicines.length < 1) return;

    // ---- Drug Interaction Check ----
    if (medicines.length >= 2) {
        const intPanel = document.getElementById("interactionPanel");
        const intResult = document.getElementById("interactionResult");
        const intBadge  = document.getElementById("interactionBadge");
        intPanel.style.display = "block";

        try {
            const resp = await fetch(`${API_BASE}/api/check-interactions`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ medicines })
            });
            const data = await resp.json();
            if (data.interactions) {
                const hasSevere = data.interactions.includes("🔴");
                intBadge.textContent  = hasSevere ? "⚠️ Warnings Found" : "✅ Checked";
                intBadge.className    = `feature-badge ${hasSevere ? "badge-danger" : "badge-ok"}`;
                intResult.innerHTML   = formatMarkdown(data.interactions);
            } else {
                intBadge.textContent  = "⚠️ Unavailable";
                intBadge.className    = "feature-badge badge-warning";
                intResult.textContent = data.warning || data.error || "Could not check interactions.";
            }
        } catch (e) {
            document.getElementById("interactionBadge").textContent = "Error";
            document.getElementById("interactionResult").textContent = e.message;
        }
    }

    // ---- Allergy Check ----
    if (userAllergies.length > 0) {
        const algPanel  = document.getElementById("allergyPanel");
        const algResult = document.getElementById("allergyResult");
        const algBadge  = document.getElementById("allergyBadge");
        algPanel.style.display = "block";

        try {
            const resp = await fetch(`${API_BASE}/api/check-allergies`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ medicines, allergies: userAllergies })
            });
            const data = await resp.json();
            if (data.allergy_report) {
                const hasRisk = data.allergy_report.includes("🔴") || data.allergy_report.includes("🟡");
                algBadge.textContent  = hasRisk ? "🚨 Alert!" : "✅ Safe";
                algBadge.className    = `feature-badge ${hasRisk ? "badge-danger" : "badge-ok"}`;
                algResult.innerHTML   = formatMarkdown(data.allergy_report);
            } else {
                algBadge.textContent  = data.message ? "✅ Safe" : "⚠️ Unavailable";
                algBadge.className    = "feature-badge badge-ok";
                algResult.textContent = data.message || data.warning || data.error || "No allergy info.";
            }
        } catch (e) {
            document.getElementById("allergyBadge").textContent = "Error";
            document.getElementById("allergyResult").textContent = e.message;
        }
    }
}

// ================================================================
// 7. Doctors App — Hospital-wise view with available specialties
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

const FACILITY_ICONS = { hospital: "🏥", clinic: "🏨", doctors: "👨‍⚕️" };

let allHospitals = [];
let doctorsLoaded = false;

// Tab switching
document.getElementById("tabScanner").addEventListener("click", () => switchTab("scanner"));
document.getElementById("tabDoctors").addEventListener("click", () => {
    switchTab("doctors");
    if (!doctorsLoaded) {
        initDoctorSlider();
        loadDoctors();
    }
});

// Radius slider setup
function initDoctorSlider() {
    const slider = document.getElementById("doctorRadiusSlider");
    const label  = document.getElementById("doctorRadiusLabel");
    if (!slider) return;

    function updateSlider() {
        const val = parseInt(slider.value);
        const pct = ((val - 1) / (20 - 1)) * 100;
        slider.style.setProperty("--slider-pct", pct + "%");
        label.textContent = val + " km";
    }
    updateSlider();

    let debounceTimer;
    slider.addEventListener("input", () => {
        updateSlider();
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            doctorsLoaded = false; // force reload
            loadDoctors();
        }, 600); // reload 600ms after user stops sliding
    });
}

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

// Load hospitals from backend
function loadDoctors() {
    if (!navigator.geolocation) {
        renderDoctorError("Geolocation not supported by your browser.");
        return;
    }

    navigator.geolocation.getCurrentPosition(async (pos) => {
        const lat = pos.coords.latitude;
        const lon = pos.coords.longitude;
        const sliderKm = parseInt(document.getElementById("doctorRadiusSlider")?.value || 5);
        const radius   = sliderKm * 1000;

        // Show loading state
        document.getElementById("doctorGrid").innerHTML = `
            <div class="doctor-loading">
                <div class="spinner" style="width:40px;height:40px;margin:0 auto 16px;"></div>
                <p>Searching within ${sliderKm} km...</p>
            </div>`;

        try {
            const response = await fetch(`${API_BASE}/api/hospitals?lat=${lat}&lon=${lon}&radius=${radius}`);
            if (!response.ok) throw new Error(`Server returned ${response.status}`);
            const data = await response.json();

            allHospitals = data.hospitals || [];
            doctorsLoaded = true;

            // Count total available specialties across all hospitals
            const totalSpecs = allHospitals.reduce((sum, h) => sum + (h.available_specialties || []).length, 0);
            document.getElementById("statTotal").textContent     = totalSpecs;
            document.getElementById("statAvailable").textContent = allHospitals.length;
            document.getElementById("statHospitals").textContent = sliderKm + " km";

            renderHospitals(allHospitals);
            setupDoctorFilters();

        } catch (err) {
            renderDoctorError(`Failed to load nearby hospitals: ${err.message}`);
        }
    }, () => {
        renderDoctorError("Location access denied. Please enable location access to find nearby hospitals.");
    });
}

// Render hospital-wise cards
function renderHospitals(hospitals) {
    const grid = document.getElementById("doctorGrid");

    if (!hospitals || hospitals.length === 0) {
        grid.innerHTML = `
            <div class="no-doctors">
                <span class="no-doctors-icon">🔍</span>
                <p>No medical facilities found nearby. Try increasing the radius.</p>
            </div>`;
        return;
    }

    grid.innerHTML = hospitals.map((h, i) => {
        const mapLink  = `https://www.google.com/maps/dir/?api=1&destination=${h.lat},${h.lon}`;
        const facIcon  = FACILITY_ICONS[h.type] || "🏥";
        const specs    = h.available_specialties || [];

        const specBadges = specs.map(spec => `
            <span style="display:inline-flex; align-items:center; gap:4px; background:rgba(139,92,246,0.12);
                color:var(--accent-purple-light); border:1px solid rgba(139,92,246,0.2);
                border-radius:20px; padding:4px 10px; font-size:0.72rem; font-weight:600;">
                ${SPECIALTIES_ICONS[spec] || "💊"} ${spec}
            </span>`).join("");

        return `
            <div class="doctor-card" style="animation-delay: ${i * 0.07}s;">
                <div class="doctor-card-top">
                    <div class="doctor-avatar" style="font-size:1.8rem;">${facIcon}</div>
                    <div class="doctor-info">
                        <div class="doctor-name">${h.name}</div>
                        <div class="doctor-specialty">${h.type.toUpperCase()}</div>
                        <div class="doctor-hospital">🕐 OPD: ${h.opd_time}</div>
                    </div>
                    <div class="facility-status-badge" style="background:#22c55e22; color:#22c55e; border:1px solid #22c55e44; white-space:nowrap;">
                        ✅ Open
                    </div>
                </div>

                <div style="padding: 12px 14px; background:rgba(0,0,0,0.2); border-radius:10px; border:1px solid rgba(255,255,255,0.05);">
                    <div style="font-size:0.72rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.06em; font-weight:600; margin-bottom:10px;">
                        ✅ Available Specialties (${specs.length})
                    </div>
                    <div style="display:flex; flex-wrap:wrap; gap:6px;">
                        ${specBadges || '<span style="color:var(--text-muted); font-size:0.8rem;">No specialties available right now</span>'}
                    </div>
                </div>

                <div class="doctor-actions">
                    <a href="tel:${h.phone}" class="btn-action btn-action-call">📞 ${h.phone}</a>
                    <a href="${mapLink}" target="_blank" class="btn-action btn-action-map">📍 Navigate</a>
                </div>
            </div>`;
    }).join("");
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
    const search    = document.getElementById("doctorSearch");
    const specFilt  = document.getElementById("specialtyFilter");

    function applyFilters() {
        const q    = search.value.toLowerCase();
        const spec = specFilt.value;

        const filtered = allHospitals.filter(h => {
            const specs = h.available_specialties || [];
            const matchQ    = !q || h.name.toLowerCase().includes(q) || specs.some(s => s.toLowerCase().includes(q));
            const matchSpec = !spec || specs.includes(spec);
            return matchQ && matchSpec;
        });

        renderHospitals(filtered);
    }

    search.addEventListener("input", applyFilters);
    specFilt.addEventListener("change", applyFilters);
}
