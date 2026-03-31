// Background script
chrome.runtime.onInstalled.addListener(() => {
    console.log("Gmail Spam Detector Extension Installed");
});

// Listen for messages if needed (e.g., for cross-origin requests if content script can't do it directly)
// Currently, content script and popup will fetch directly from localhost.
