import { sendRelevantDataToExtension, fetchResponseFromExtension, getResponseFromExtension } from './call.js';
import { query } from './main.js';

// Function to display a message with effect
function displayMessageWithEffect(containerId, messageContent, type, link = '', callback) {
    const chatWindow = document.getElementById(containerId);
    
    if (!chatWindow) {
        console.error(`Element with ID '${containerId}' not found.`);
        return;
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', type);

    if (type === 'bot-message') {
        messageContent = `<strong>Bot:</strong><br>${messageContent}`;
    } else if (type === 'user-message') {
        messageContent = `<strong>Me:</strong><br>${messageContent}`;
    } else if (type === 'error-message') {
        messageContent = `<strong>Error:</strong><br>${messageContent}`;
    }

    if (link) {
        messageContent += `<br><a href="${link}" target="_blank">Read more</a><br><button class="relevant-button">Relevant</button>`;
    }

    chatWindow.prepend(messageDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;

    let words = messageContent.split(' ');
    let index = 0;
    let content = '';

    function addWord() {
        if (index < words.length) {
            content += words[index] + ' ';
            messageDiv.innerHTML = content;
            chatWindow.scrollTop = chatWindow.scrollHeight;
            index++;
            setTimeout(addWord, 100); // Adjust the interval as needed
        } else {
            if (link) {
                const relevantButton = messageDiv.querySelector('.relevant-button');
                if (relevantButton) {
                    relevantButton.addEventListener('click', () => sendRelevantDataToExtension(query, messageContent, link));
                    relevantButton.addEventListener('click', (event) => showPopupMessage(event, 'Thank you for your feedback!'));
                }
            }
            if (callback) {
                callback();
            }
        }
    }

    addWord();
}

// Function to display initial bot message
export function displayInitialBotMessage() {
    const message = "Hi, I am here to rescue";
    displayMessageWithEffect('display', message, 'bot-message');
}

// Function to display the user's message
export function displayUserMessage(message) {
    displayMessageWithEffect('display', message, 'user-message');
}

// Function to display the bot's response
export async function displayBotMessage(data) {
    displayMessageWithEffect('display', data, 'bot-message', '', async () => {
        await getResponseFromExtension();
    });
}

// Function to display an error message
export function displayErrorMessage(errorMsg) {
    displayMessageWithEffect('display', errorMsg, 'error-message');
}

// Function to display link and answer
export async function display_answer_url(answer, link = '') {
    displayMessageWithEffect('display', answer, 'bot-message', link);
}

// Function to show spinner
export function showSpinner() {
    const chatWindow = document.getElementById('display');
    if (!chatWindow) {
        console.error("Element with ID 'display' not found.");
        return;
    }

    const spinnerDiv = document.createElement('div');
    spinnerDiv.id = 'spinner';
    spinnerDiv.classList.add('spinner');
    spinnerDiv.innerHTML = `<div class="loader"></div>`;
    chatWindow.prepend(spinnerDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Function to hide spinner
export function hideSpinner() {
    const spinnerDiv = document.getElementById('spinner');
    if (spinnerDiv) {
        spinnerDiv.remove();
    }
}

// Function to show popup message
export function showPopupMessage(event, message) {
    let popup = document.createElement('div');
    popup.classList.add('popup-message');
    popup.textContent = message;

    document.body.appendChild(popup);

    let buttonRect = event.target.getBoundingClientRect();
    popup.style.top = `${buttonRect.bottom + window.scrollY + 10}px`;
    popup.style.left = `${buttonRect.left + window.scrollX}px`;

    popup.style.visibility = 'visible';

    setTimeout(() => {
        popup.style.visibility = 'hidden';
        document.body.removeChild(popup);
    }, 3000);
}
