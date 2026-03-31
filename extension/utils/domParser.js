const DomParser = {
    getSender: () => {
        const senderElement = document.querySelector('.gD');
        return senderElement ? senderElement.getAttribute('email') : null;
    },

    getSubject: () => {
        const subjectElement = document.querySelector('.hP');
        return subjectElement ? subjectElement.innerText : null;
    },

    getBody: () => {
        const bodyElement = document.querySelector('.a3s.aiL');
        return bodyElement ? bodyElement.innerText : null;
    }
};

// Export for use in content.js (if using modules, but for simple content script we can just include it or copy it)
// For this project structure, we'll just import it or paste it in content.js if we don't use a bundler.
// Since we are not using a bundler, I will include this logic directly in content.js or load it as a separate file in manifest.
