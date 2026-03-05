document.addEventListener("DOMContentLoaded", () => {
    // ── Initialize Icons ──
    lucide.createIcons();

    // ── DOM Elements ──
    const video = document.getElementById("webcam");
    const canvasOverlay = document.getElementById("overlay");
    const ctx = canvasOverlay.getContext("2d");
    const startBtn = document.getElementById("start-btn");
    const stopBtn = document.getElementById("stop-btn");
    const placeholder = document.getElementById("camera-placeholder");
    const videoWrapper = document.getElementById("video-wrapper");
    const statusBadge = document.getElementById("camera-status");
    const statusDot = statusBadge.querySelector(".dot");
    const statusText = statusBadge.querySelector(".status-text");
    const themeBtn = document.getElementById("theme-toggle");

    const phoneWarn = document.getElementById("phone-warning-banner");
    const socialWarn = document.getElementById("social-warning-banner");
    const wasteWarn = document.getElementById("waste-warning-banner");
    const liveState = document.getElementById("live-state-value");

    const emotionValue = document.getElementById("emotion-value");
    const confidenceValue = document.getElementById("confidence-value");
    const confidenceFill = document.getElementById("confidence-fill");
    const emotionBars = document.getElementById("emotion-bars");
    const historyLog = document.getElementById("history-log");
    const predLoader = document.getElementById("prediction-loader");
    const predStatus = document.getElementById("prediction-status");

    // Video Analysis
    const tabLive = document.getElementById("tab-live");
    const tabVideo = document.getElementById("tab-video");
    const liveView = document.getElementById("live-view");
    const videoView = document.getElementById("video-view");

    const uploadZone = document.getElementById("upload-zone");
    const videoInput = document.getElementById("video-upload-input");
    const browseBtn = document.getElementById("browse-btn");
    const videoPreviewWrap = document.getElementById("video-preview-wrapper");
    const videoPreview = document.getElementById("uploaded-video-preview");
    const analyzeVideoBtn = document.getElementById("analyze-video-btn");

    const videoPredLoader = document.getElementById("video-prediction-loader");
    const videoPredStatus = document.getElementById("video-prediction-status");
    const effScoreValue = document.getElementById("efficiency-score-value");
    const videoEmotionBars = document.getElementById("video-emotion-bars");
    const totalFramesText = document.getElementById("total-frames-analyzed");
    const scoreRingFill = document.getElementById("score-ring-fill");

    // Extra stat elements (new HTML)
    const statFrames = document.getElementById("stat-frames");
    const statPhone = document.getElementById("stat-phone");
    const statChat = document.getElementById("stat-chat");

    // ── State ──
    let stream = null;
    let captureInterval = null;
    const FPS = 2;
    let isProcessing = false;
    let currentTheme = 'dark';
    let lastEmotionForHistory = null;

    // Hidden canvas for frame extraction
    const captureCanvas = document.createElement("canvas");
    const captureCtx = captureCanvas.getContext("2d");

    // All emotion keys (including disgust)
    const emotionKeys = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];

    const emotionEmoji = {
        happy: '😊', sad: '😢', angry: '😠', fear: '😨',
        surprise: '😲', neutral: '😐', disgust: '🤢'
    };

    // ── Ring progress circumference = 2π×30 ≈ 188.5 ──
    const RING_CIRCUMFERENCE = 2 * Math.PI * 30; // ~188.5

    function setRing(percent) {
        if (!scoreRingFill) return;
        const offset = RING_CIRCUMFERENCE - (percent / 100) * RING_CIRCUMFERENCE;
        scoreRingFill.style.strokeDasharray = RING_CIRCUMFERENCE;
        scoreRingFill.style.strokeDashoffset = Math.max(0, offset);
    }

    // ── Theme Toggle ──
    themeBtn.addEventListener("click", () => {
        const root = document.documentElement;
        const icon = document.getElementById("theme-icon");
        if (currentTheme === 'dark') {
            currentTheme = 'light';
            root.setAttribute("data-theme", "light");
            icon.setAttribute("data-lucide", "moon");
        } else {
            currentTheme = 'dark';
            root.setAttribute("data-theme", "dark");
            icon.setAttribute("data-lucide", "sun");
        }
        lucide.createIcons();
    });

    // ── Start Camera ──
    startBtn.addEventListener("click", async () => {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert(
                "Camera API is blocked by your browser! 🔒\n\n" +
                "Browsers block camera access on non-secure connections.\n" +
                "• If accessing from the same PC, use EXACTLY: http://localhost:5000\n" +
                "• For remote access you must set up HTTPS."
            );
            return;
        }

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: "user" },
                audio: false
            });

            video.srcObject = stream;

            video.onloadedmetadata = () => {
                video.play();

                canvasOverlay.width = video.videoWidth;
                canvasOverlay.height = video.videoHeight;
                captureCanvas.width = video.videoWidth;
                captureCanvas.height = video.videoHeight;

                placeholder.classList.add("hidden");
                videoWrapper.classList.remove("hidden");

                statusDot.classList.replace("disconnected", "connected");
                statusText.innerText = "Connected";
                statusBadge.classList.add("connected-badge");

                predStatus.innerText = "Analyzing stream…";
                liveState.innerText = "Initializing…";
                liveState.classList.remove("warning");

                captureInterval = setInterval(processFrame, 1000 / FPS);
            };
        } catch (err) {
            console.error("Camera error:", err);
            let msg = "Could not access camera. ";
            if (err.name === 'NotAllowedError') msg += "Permission denied. Allow camera in your browser.";
            else if (err.name === 'NotFoundError') msg += "No camera found on this device.";
            else if (err.name === 'NotReadableError') msg += "Camera is already in use.";
            else if (window.location.protocol !== 'https:' &&
                window.location.hostname !== 'localhost' &&
                window.location.hostname !== '127.0.0.1') {
                msg += "Requires HTTPS or localhost.";
            }
            alert(msg + "\n\nDetails: " + err.message);
        }
    });

    // ── Stop Camera ──
    stopBtn.addEventListener("click", stopCamera);

    function stopCamera() {
        if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
        if (captureInterval) { clearInterval(captureInterval); captureInterval = null; }

        videoWrapper.classList.add("hidden");
        placeholder.classList.remove("hidden");
        statusDot.classList.replace("connected", "disconnected");
        statusText.innerText = "Disconnected";
        statusBadge.classList.remove("connected-badge");
        predStatus.innerText = "Waiting for face…";

        clearCanvas();
        resetUI();
    }

    // ── Process Frame ──
    async function processFrame() {
        if (isProcessing) return;
        if (video.paused || video.ended) return;

        captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
        const base64 = captureCanvas.toDataURL("image/jpeg", 0.85);

        isProcessing = true;
        predLoader.classList.remove("hidden");

        try {
            // NOTE: For Netlify deployment, change this to your hosted backend URL.
            // Example: const API_URL = "https://your-backend-app.onrender.com/predict_emotion";
            const API_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
                ? '/predict_emotion'
                : 'https://YOUR_BACKEND_URL.onrender.com/predict_emotion';

            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64 })
            });

            const result = await response.json();

            if (!response.ok) {
                predStatus.innerText = result.error || "Analyzing…";
                clearCanvas();
                if (result.error && result.error.includes("face")) {
                    clearUIWhenNoFace();
                }
            } else {
                updateUI(result);
            }
        } catch (err) {
            console.error("API Error:", err);
            predStatus.innerText = "Network Error";
        } finally {
            isProcessing = false;
            predLoader.classList.add("hidden");
        }
    }

    // ── Update UI ──
    function updateUI(data) {
        const { emotion, confidence, box, all_scores,
            phone_detected, phone_boxes,
            socializing_detected, person_boxes,
            waste_detected, waste_boxes,
            laptop_detected, laptop_boxes } = data;

        predStatus.innerText = "✓ Face Detected";

        emotionValue.innerText = (emotionEmoji[emotion] || '') + ' ' + (emotion || '--');
        confidenceValue.innerText = confidence || '--';
        confidenceFill.style.width = `${confidence || 0}%`;

        // Hide all banners first
        phoneWarn.classList.add("hidden");
        socialWarn.classList.add("hidden");
        wasteWarn.classList.add("hidden");

        if (phone_detected) {
            phoneWarn.classList.remove("hidden");
            liveState.innerText = "Distracted / On Phone";
            liveState.classList.add("warning");
        } else if (socializing_detected) {
            socialWarn.classList.remove("hidden");
            liveState.innerText = "Chatting / Socializing";
            liveState.classList.add("warning");
        } else if (waste_detected) {
            wasteWarn.classList.remove("hidden");
            liveState.innerText = "Break Time / Eating";
            liveState.classList.add("warning");
        } else if (laptop_detected) {
            liveState.innerText = "💻 Working on Laptop";
            liveState.classList.remove("warning");
        } else {
            if (['happy', 'neutral'].includes(emotion)) {
                liveState.innerText = "✅ Focusing / Engaged";
                liveState.classList.remove("warning");
            } else {
                liveState.innerText = "⚠️ Stressed / Disengaged";
                liveState.classList.add("warning");
            }
        }

        drawBoundingBoxes(box, emotion, confidence, phone_boxes, person_boxes, waste_boxes, laptop_boxes);
        updateBarChart(emotionBars, all_scores, 'percent');
        addToHistory(emotion, confidence);
    }

    function clearUIWhenNoFace() {
        emotionValue.innerText = "--";
        confidenceValue.innerText = "--";
        confidenceFill.style.width = "0%";
        phoneWarn.classList.add("hidden");
        socialWarn.classList.add("hidden");
        wasteWarn.classList.add("hidden");
        liveState.innerText = "Away from Desk";
        liveState.classList.add("warning");
        resetBarChart(emotionBars);
    }

    // ── Drawing ──
    function drawBoundingBoxes(faceBox, emotion, confidence, phoneBoxes, personBoxes, wasteBoxes, laptopBoxes) {
        clearCanvas();
        const W = canvasOverlay.width;

        // Face box
        if (faceBox && Object.keys(faceBox).length > 0) {
            const { x, y, w, h } = faceBox;
            const mx = W - x - w;

            ctx.strokeStyle = getEmotionColor(emotion);
            ctx.lineWidth = 2.5;
            ctx.strokeRect(mx, y, w, h);

            const label = `${(emotionEmoji[emotion] || '')} ${emotion.toUpperCase()} ${confidence}%`;
            ctx.font = "bold 13px 'Inter', sans-serif";
            const tw = ctx.measureText(label).width;

            ctx.fillStyle = getEmotionColor(emotion);
            ctx.fillRect(mx, y - 26, tw + 12, 26);
            ctx.fillStyle = "#ffffff";
            ctx.fillText(label, mx + 6, y - 8);
        }

        // Phone boxes
        if (phoneBoxes && phoneBoxes.length > 0) {
            phoneBoxes.forEach(pb => {
                const mx = W - pb.x - pb.w;
                ctx.strokeStyle = "#ef4444";
                ctx.lineWidth = 3;
                ctx.strokeRect(mx, pb.y, pb.w, pb.h);

                const label = `📵 PHONE ${pb.conf ? pb.conf + '%' : ''}`;
                ctx.font = "bold 12px 'Inter', sans-serif";
                const tw = ctx.measureText(label).width;
                ctx.fillStyle = "#ef4444";
                ctx.fillRect(mx, pb.y - 24, tw + 10, 24);
                ctx.fillStyle = "#fff";
                ctx.fillText(label, mx + 5, pb.y - 7);
            });
        }

        // Person boxes
        if (personBoxes && personBoxes.length > 0) {
            ctx.strokeStyle = "#f97316";
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 4]);
            personBoxes.forEach(pb => {
                const mx = W - pb.x - pb.w;
                ctx.strokeRect(mx, pb.y, pb.w, pb.h);
                ctx.setLineDash([]);
                const label = "👤 PERSON";
                ctx.font = "bold 12px 'Inter', sans-serif";
                const tw = ctx.measureText(label).width;
                ctx.fillStyle = "#f97316";
                ctx.fillRect(mx, pb.y - 22, tw + 8, 22);
                ctx.fillStyle = "#fff";
                ctx.fillText(label, mx + 4, pb.y - 6);
                ctx.setLineDash([6, 4]);
            });
            ctx.setLineDash([]);
        }

        // Waste boxes
        if (wasteBoxes && wasteBoxes.length > 0) {
            ctx.strokeStyle = "#f59e0b";
            ctx.lineWidth = 2.5;
            wasteBoxes.forEach(wb => {
                const mx = W - wb.x - wb.w;
                ctx.strokeRect(mx, wb.y, wb.w, wb.h);
                const label = `☕ ${wb.label || 'ITEM'}`;
                ctx.font = "bold 12px 'Inter', sans-serif";
                const tw = ctx.measureText(label).width;
                ctx.fillStyle = "#f59e0b";
                ctx.fillRect(mx, wb.y - 22, tw + 8, 22);
                ctx.fillStyle = "#fff";
                ctx.fillText(label, mx + 4, wb.y - 6);
            });
        }

        // Laptop boxes
        if (laptopBoxes && laptopBoxes.length > 0) {
            ctx.strokeStyle = "#10b981";
            ctx.lineWidth = 2.5;
            laptopBoxes.forEach(lb => {
                const mx = W - lb.x - lb.w;
                ctx.strokeRect(mx, lb.y, lb.w, lb.h);
                const label = "💻 LAPTOP";
                ctx.font = "bold 12px 'Inter', sans-serif";
                const tw = ctx.measureText(label).width;
                ctx.fillStyle = "#10b981";
                ctx.fillRect(mx, lb.y - 22, tw + 8, 22);
                ctx.fillStyle = "#fff";
                ctx.fillText(label, mx + 4, lb.y - 6);
            });
        }
    }

    function clearCanvas() {
        ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);
    }

    function getEmotionColor(emotion) {
        const map = {
            happy: "#22c55e", sad: "#60a5fa", angry: "#ef4444",
            fear: "#a78bfa", surprise: "#fbbf24", neutral: "#94a3b8",
            disgust: "#f97316"
        };
        return map[emotion] || "#ffffff";
    }

    // ── Bar Chart ──
    function updateBarChart(container, scores, mode) {
        container.innerHTML = '';
        emotionKeys.forEach(emo => {
            const val = scores ? (scores[emo] || 0) : 0;
            const pct = mode === 'percent' ? val.toFixed(1) : val;
            const width = mode === 'percent' ? val.toFixed(1) : 0;

            const row = document.createElement("div");
            row.className = "bar-row";
            row.innerHTML = `
                <span class="bar-label">
                    <span class="emo-dot" style="background:${getEmotionColor(emo)}"></span>
                    ${emotionEmoji[emo] || ''} ${emo}
                </span>
                <div class="bar-track">
                    <div class="bar-fill ${emo}" style="width:${width}%"></div>
                </div>
                <span class="bar-value">${pct}%</span>
            `;
            container.appendChild(row);
        });
    }

    function resetBarChart(container) {
        if (!container) return;
        container.innerHTML = '';
        emotionKeys.forEach(emo => {
            const row = document.createElement("div");
            row.className = "bar-row";
            row.innerHTML = `
                <span class="bar-label">
                    <span class="emo-dot" style="background:${getEmotionColor(emo)}"></span>
                    ${emotionEmoji[emo] || ''} ${emo}
                </span>
                <div class="bar-track"><div class="bar-fill ${emo}" style="width:0%"></div></div>
                <span class="bar-value">0%</span>
            `;
            container.appendChild(row);
        });
    }

    // ── History Log ──
    function addToHistory(emotion, confidence) {
        if (!emotion || emotion === '--') return;
        if (emotion === lastEmotionForHistory) return;
        lastEmotionForHistory = emotion;

        const li = document.createElement("li");
        li.className = `history-item ${emotion}`;
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        li.innerHTML = `
            <span class="history-emotion">${emotionEmoji[emotion] || ''} ${emotion}</span>
            <span class="history-time">${time}</span>
        `;
        historyLog.insertBefore(li, historyLog.firstChild);

        // Keep max 20 items
        while (historyLog.children.length > 20) {
            historyLog.removeChild(historyLog.lastChild);
        }
    }

    function resetUI() {
        emotionValue.innerText = "--";
        confidenceValue.innerText = "--";
        confidenceFill.style.width = "0%";
        liveState.innerText = "Idle";
        liveState.classList.remove("warning");
        phoneWarn.classList.add("hidden");
        socialWarn.classList.add("hidden");
        wasteWarn.classList.add("hidden");
        resetBarChart(emotionBars);
        lastEmotionForHistory = null;
    }

    // ── Tab Switching ──
    tabLive.addEventListener("click", () => {
        tabLive.classList.add("active");
        tabVideo.classList.remove("active");
        liveView.classList.remove("hidden");
        videoView.classList.add("hidden");
        lucide.createIcons();
        if (!stream) startBtn.click();
    });

    tabVideo.addEventListener("click", () => {
        tabVideo.classList.add("active");
        tabLive.classList.remove("active");
        videoView.classList.remove("hidden");
        liveView.classList.add("hidden");
        lucide.createIcons();
        stopCamera();
    });

    // ── Upload ──
    browseBtn.addEventListener("click", () => videoInput.click());

    uploadZone.addEventListener("dragover", e => { e.preventDefault(); uploadZone.classList.add("dragover"); });
    uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
    uploadZone.addEventListener("drop", e => {
        e.preventDefault();
        uploadZone.classList.remove("dragover");
        if (e.dataTransfer.files.length) {
            videoInput.files = e.dataTransfer.files;
            handleVideoSelection();
        }
    });

    videoInput.addEventListener("change", handleVideoSelection);

    let selectedFile = null;

    function handleVideoSelection() {
        if (!videoInput.files.length) return;
        selectedFile = videoInput.files[0];
        videoPreview.src = URL.createObjectURL(selectedFile);
        uploadZone.classList.add("hidden");
        videoPreviewWrap.classList.remove("hidden");

        effScoreValue.innerText = "--";
        totalFramesText.innerText = "0";
        videoEmotionBars.innerHTML = "";
        videoPredStatus.innerText = "Video ready. Click Analyze.";
        videoPredLoader.classList.add("hidden");
        if (statFrames) statFrames.innerText = "0";
        if (statPhone) statPhone.innerText = "0";
        if (statChat) statChat.innerText = "0";
        if (document.getElementById("stat-laptop")) document.getElementById("stat-laptop").innerText = "0";
        setRing(0);
    }

    // ── Analyze Video ──
    analyzeVideoBtn.addEventListener("click", async () => {
        if (!selectedFile) return;

        videoPredStatus.innerText = "Uploading & analyzing… (may take a moment)";
        videoPredLoader.classList.remove("hidden");
        analyzeVideoBtn.disabled = true;

        const formData = new FormData();
        formData.append("video", selectedFile);

        try {
            const response = await fetch('/analyze_video', { method: 'POST', body: formData });
            const result = await response.json();

            if (!response.ok) {
                videoPredStatus.innerText = result.error || "Analysis failed.";
            } else {
                displayVideoResults(result);
            }
        } catch (err) {
            console.error("Video analysis error:", err);
            videoPredStatus.innerText = "Network Error.";
        } finally {
            videoPredLoader.classList.add("hidden");
            analyzeVideoBtn.disabled = false;
        }
    });

    function displayVideoResults(data) {
        const { emotions, analyzed_frames, efficiency_score,
            phone_frames, socializing_frames, laptop_frames } = data;

        videoPredStatus.innerText = "✅ Analysis Complete!";
        effScoreValue.innerText = efficiency_score;
        totalFramesText.innerText = analyzed_frames;

        // Update extra stats
        if (statFrames) statFrames.innerText = analyzed_frames || 0;
        if (statPhone) statPhone.innerText = phone_frames || 0;
        if (statChat) statChat.innerText = socializing_frames || 0;
        if (document.getElementById("stat-laptop")) {
            document.getElementById("stat-laptop").innerText = laptop_frames || 0;
        }

        // Animate ring
        setRing(efficiency_score || 0);

        // Build bar chart
        videoEmotionBars.innerHTML = '';
        let totalEmo = 0;
        emotionKeys.forEach(e => { totalEmo += (emotions[e] || 0); });

        emotionKeys.forEach(emo => {
            const count = emotions[emo] || 0;
            const pct = totalEmo === 0 ? 0 : (count / totalEmo * 100);

            const row = document.createElement("div");
            row.className = "bar-row";
            row.innerHTML = `
                <span class="bar-label">
                    <span class="emo-dot" style="background:${getEmotionColor(emo)}"></span>
                    ${emotionEmoji[emo] || ''} ${emo}
                </span>
                <div class="bar-track">
                    <div class="bar-fill ${emo}" style="width:${pct.toFixed(1)}%"></div>
                </div>
                <span class="bar-value">${count}</span>
            `;
            videoEmotionBars.appendChild(row);
        });
    }

    // ── Init ──
    resetBarChart(emotionBars);
    setRing(0);

    // Auto-start camera
    setTimeout(() => startBtn.click(), 500);
});
