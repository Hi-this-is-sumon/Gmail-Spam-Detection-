document.addEventListener('DOMContentLoaded', () => {
    const btnGet = document.getElementById('btn-get');
    const btnAnalyze = document.getElementById('btn-analyze');
    const emailInput = document.getElementById('email-input');
    const resultBox = document.getElementById('result');
    const resultContent = document.getElementById('result-content');
    const loaderContainer = document.querySelector('.loader-container');

    // 1. Get from Gmail
    btnGet.addEventListener('click', async () => {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

        chrome.tabs.sendMessage(tab.id, { command: "get_email_data" }, (response) => {
            if (chrome.runtime.lastError) {
                alert("Could not connect to the page. Make sure you are on a Gmail email.");
                return;
            }
            if (response) {
                // Combine all email data into single input
                const emailText = `From: ${response.sender || ''}\nSubject: ${response.subject || ''}\n\n${response.body || ''}`;
                emailInput.value = emailText;

                // Animate the textarea
                emailInput.style.animation = 'none';
                setTimeout(() => {
                    emailInput.style.animation = 'textareaGrow 0.4s cubic-bezier(0.4, 0, 0.2, 1)';
                }, 10);
            } else {
                alert("No email data found. Open an email first.");
            }
        });
    });

    // 2. Analyze
    btnAnalyze.addEventListener('click', async () => {
        const emailText = emailInput.value;

        if (!emailText.trim()) {
            alert("Please paste email content or get it from Gmail.");
            return;
        }

        // Parse email text to extract sender, subject, body
        const { sender, subject, body } = parseEmail(emailText);

        // UI State: Loading
        resultBox.classList.remove('hidden', 'visible', 'spam', 'safe', 'whitelisted');
        resultBox.classList.add('visible');
        resultContent.classList.add('hidden');
        loaderContainer.classList.remove('hidden');

        try {
            const response = await fetch("http://localhost:8000/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sender, subject, body })
            });

            const data = await response.json();


            // UI State: Result
            loaderContainer.classList.add('hidden');
            resultContent.classList.remove('hidden');

            // Normalize label for CSS class and display
            let cssClass;
            let displayLabel;

            if (data.label === 'Spam') {
                cssClass = 'spam';
                displayLabel = 'SPAM';
            } else if (data.label === 'Not Spam') {
                cssClass = 'safe';
                displayLabel = 'NOT SPAM';
            } else if (data.label === 'whitelisted') {
                cssClass = 'whitelisted';
                displayLabel = 'WHITELISTED';
            } else {
                cssClass = 'safe';
                displayLabel = data.label.toUpperCase();
            }

            document.getElementById('label').innerText = displayLabel;
            document.getElementById('confidence').innerText = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
            document.getElementById('reason').innerText = data.reason;

            // Animate confidence bar
            const confidenceFill = document.getElementById('confidence-fill');
            setTimeout(() => {
                confidenceFill.style.width = `${data.confidence * 100}%`;
            }, 100);

            resultBox.classList.add(cssClass);

        } catch (error) {
            console.error(error);
            loaderContainer.classList.add('hidden');
            resultContent.classList.remove('hidden');
            document.getElementById('label').innerText = "ERROR";
            document.getElementById('reason').innerText = "Could not connect to backend. Make sure the server is running.";
            // Reset confidence bar
            document.getElementById('confidence-fill').style.width = '0%';
            document.getElementById('confidence').innerText = '';
        }
    });

    // Helper function to parse email text
    function parseEmail(text) {
        let sender = '';
        let subject = '';
        let body = text;

        // Try to extract From: line
        const fromMatch = text.match(/From:\s*(.+)/i);
        if (fromMatch) {
            sender = fromMatch[1].trim();
        }

        // Try to extract Subject: line
        const subjectMatch = text.match(/Subject:\s*(.+)/i);
        if (subjectMatch) {
            subject = subjectMatch[1].trim();
        }

        // Extract body (everything after From/Subject or entire text)
        if (fromMatch || subjectMatch) {
            // Remove From and Subject lines from body
            body = text.replace(/From:\s*.+/i, '')
                .replace(/Subject:\s*.+/i, '')
                .trim();
        }

        return { sender, subject, body };
    }

    // Add closing animation on window unload (optional)
    window.addEventListener('beforeunload', () => {
        document.body.classList.add('closing');
    });
});
