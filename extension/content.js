// content.js

// Utility functions (inlined for simplicity without bundler)
const getEmailData = () => {
    const senderElement = document.querySelector('.gD');
    const subjectElement = document.querySelector('.hP');
    const bodyElement = document.querySelector('.a3s.aiL');

    return {
        sender: senderElement ? senderElement.getAttribute('email') : null,
        subject: subjectElement ? subjectElement.innerText : null,
        body: bodyElement ? bodyElement.innerText : null
    };
};

const injectBanner = (data, prediction) => {
    // Remove existing banner
    const existingBanner = document.getElementById('spam-detector-banner');
    if (existingBanner) existingBanner.remove();

    const banner = document.createElement('div');
    banner.id = 'spam-detector-banner';

    // Styles
    banner.style.width = '100%';
    banner.style.padding = '15px';
    banner.style.marginBottom = '10px';
    banner.style.fontFamily = 'Arial, sans-serif';
    banner.style.fontWeight = 'bold';
    banner.style.display = 'flex';
    banner.style.alignItems = 'center';
    banner.style.justifyContent = 'space-between';
    banner.style.boxShadow = '0 2px 5px rgba(0,0,0,0.1)';
    banner.style.transition = 'all 0.5s ease-in-out';
    banner.style.zIndex = '9999';

    if (prediction.label === 'Spam') {
        banner.style.backgroundColor = '#ffebee';
        banner.style.borderLeft = '5px solid #f44336';
        banner.style.color = '#c62828';
        banner.innerHTML = `
            <div>
                <span style="font-size: 1.2em; margin-right: 10px;">🚨</span>
                WARNING: Likely SPAM (Confidence: ${(prediction.confidence * 100).toFixed(1)}%)
                <div style="font-weight: normal; font-size: 0.9em; margin-top: 5px;">${prediction.reason}</div>
            </div>
        `;
    } else if (prediction.label === 'whitelisted') {
        banner.style.backgroundColor = '#e3f2fd';
        banner.style.borderLeft = '5px solid #2196f3';
        banner.style.color = '#1565c0';
        banner.innerHTML = `
            <div>
                <span style="font-size: 1.2em; margin-right: 10px;">🛡️</span>
                SAFE: Whitelisted Sender
                <div style="font-weight: normal; font-size: 0.9em; margin-top: 5px;">${prediction.reason}</div>
            </div>
        `;
    } else {
        banner.style.backgroundColor = '#e8f5e9';
        banner.style.borderLeft = '5px solid #4caf50';
        banner.style.color = '#2e7d32';
        banner.innerHTML = `
            <div>
                <span style="font-size: 1.2em; margin-right: 10px;">✅</span>
                SAFE: No spam detected (Confidence: ${(prediction.confidence * 100).toFixed(1)}%)
            </div>
        `;
    }

    // Inject above the email body
    const emailContainer = document.querySelector('.a3s.aiL').parentElement;
    if (emailContainer) {
        emailContainer.insertBefore(banner, emailContainer.firstChild);
    }
};

const analyzeEmail = async () => {
    const data = getEmailData();
    if (!data.sender || !data.body) return;

    try {
        const response = await fetch("http://localhost:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });
        const prediction = await response.json();
        injectBanner(data, prediction);
    } catch (error) {
        console.error("Error analyzing email:", error);
    }
};

// Observer to detect when an email is opened
const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
        if (mutation.addedNodes.length) {
            // Check if email body is present
            if (document.querySelector('.a3s.aiL')) {
                // Debounce or check if already analyzed
                if (!document.getElementById('spam-detector-banner')) {
                    setTimeout(analyzeEmail, 1000); // Wait a bit for full load
                }
            }
        }
    }
});

observer.observe(document.body, {
    childList: true,
    subtree: true
});

// Listen for messages from Popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.command === "get_email_data") {
        sendResponse(getEmailData());
    }
});
