const form = document.getElementById("action-form");
const promptInput = document.getElementById("prompt");
const gifContainer = document.getElementById("gif-container");

function showLoader() {
    gifContainer.innerHTML = '<div id="loader">⏳ Executing...</div>';
}

function showError(msg) {
    gifContainer.innerHTML = '<div style="color:#b93b2e;margin-top:1em;">' +
        (msg ?? "There was an error running your prompt.") +
        '</div>';
}

function showResult(gifUrl, videoUrl, resultText, status) {
    let html = "";
    if (status) {
        html += `<div style="margin-bottom:8px;font-weight:bold;color:${status === 'success' ? '#33754d' : '#a01716'}">${status.toUpperCase()}</div>`;
    }
    if (videoUrl) {
        html += `<video controls style="max-width:90%;border-radius:8px;box-shadow:0 1px 9px #aac1de33;margin-bottom:14px;">
            <source src="${videoUrl}" type="video/webm">
            Your browser does not support the video tag.
        </video>`;
    } else if (gifUrl) {
        html += `<img src="${gifUrl}" alt="Result GIF" style="max-width:90%;border-radius:8px;box-shadow:0 1px 9px #aac1de33;margin-bottom:14px;">`;
    }
    if (resultText) {
        html += `<div style="margin-top:6px;color:#234;margin-bottom:8px;font-size:1.07rem;">${resultText}</div>`;
    }
    gifContainer.innerHTML = html;
}

form.addEventListener("submit", async function (e) {
    e.preventDefault();
    const prompt = promptInput.value.trim();
    if (!prompt) return;
    showLoader();
    try {
        const resp = await fetch("/run", {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt })
        });
        const data = await resp.json();
        if (data.status === 'success') {
            showResult(data.gif_url, data.video_url, data.result, data.test_status);
        } else {
            showError(data.message || "Unknown error.");
        }
    } catch (err) {
        showError("Server/network error.");
        console.error(err);
    }
});
