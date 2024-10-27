function preprocessWordTimesExact(fragments) {
    // Prepare a lookup table with start and end times, and corresponding text
    const lookupTable = [];

    fragments.forEach(fragment => {
        const { begin, end, text } = fragment;
        lookupTable.push({
            start: parseFloat(begin) * 1000,  // Convert to milliseconds
            end: parseFloat(end) * 1000,      // Convert to milliseconds
            text             // Combine lines to form the complete text
        });
    });

    return lookupTable;
}

function initializeSynchronization(speechMarksFile) {
    const serverAddress = localStorage.getItem('serverAddress') || 'http://localhost:5000';
    const audioElement = document.getElementById('audio');
    const textDisplay = document.getElementById('text-display');
    const playbackRateSelector = document.getElementById('playback-rate');
    const skipBackwardButton = document.getElementById('skip-backward');
    const skipForwardButton = document.getElementById('skip-forward');

    const audio = new MediaElementPlayer(audioElement, {
        success: function(mediaElement, originalNode) {
            mediaElement.addEventListener('timeupdate', syncText);
            mediaElement.addEventListener('ratechange', resetText);
            mediaElement.addEventListener('seeked', handleSeek);
        }
    });

    let lookupTable = [];
    let currentFragmentIndex = 0;

    fetch(`${serverAddress}${speechMarksFile}`)
        .then(response => response.json())
        .then(data => {
            lookupTable = preprocessWordTimesExact(data.fragments);  // Using modified function for Gentle
            renderWordsForCurrentTime(audio.media.currentTime * 1000); // Render initial words
        })
        .catch(error => {
            console.error('Error loading word times:', error);
        });

    function syncText() {
        const currentTime = audio.media.currentTime * 1000; // Convert to milliseconds
        renderWordsForCurrentTime(currentTime);
    }

    function resetText() {
        textDisplay.innerHTML = ''; // Clear text
        renderWordsForCurrentTime(audiop.media.currentTime * 1000);
    }

    function handleSeek() {
        const currentTime = audio.media.currentTime * 1000; // Convert to milliseconds
        textDisplay.innerHTML = ''; // Clear text
        renderWordsForCurrentTime(currentTime);
    }

    function renderWordsForCurrentTime(currentTime) {
        // Find the appropriate fragment for the current time
        let fragment = null;

        // Check if the current fragment is still valid
        if (lookupTable[currentFragmentIndex] &&
            currentTime >= lookupTable[currentFragmentIndex].start &&
            currentTime <= lookupTable[currentFragmentIndex].end) {
            fragment = lookupTable[currentFragmentIndex];
        } else {
            // Otherwise, find the correct fragment from scratch
            for (let i = 0; i < lookupTable.length; i++) {
                if (currentTime >= lookupTable[i].start && currentTime <= lookupTable[i].end) {
                    fragment = lookupTable[i];
                    currentFragmentIndex = i;
                    break;
                }
            }
        }

        // Render the text if a matching fragment was found
        if (fragment) {
            renderVisibleWords(fragment.text, textDisplay);
        }
    }

    function renderVisibleWords(text, container) {
        container.innerHTML = ''; // Clear previous content

        // Create a DocumentFragment to minimize reflows
        const fragment = document.createDocumentFragment();

        // Render the text as a single string
        const textNode = document.createTextNode(text);
        fragment.appendChild(textNode);

        container.appendChild(fragment);
    }

    playbackRateSelector.addEventListener('change', function() {
        audio.media.playbackRate = parseFloat(this.value);
    });

    skipBackwardButton.addEventListener('click', function() {
        audio.media.setCurrentTime(Math.max(0, audio.media.currentTime - 5));
        handleSeek();
    });

    skipForwardButton.addEventListener('click', function() {
        audio.media.setCurrentTime(Math.min(audio.media.duration, audio.media.currentTime + 5));
        handleSeek();
    });
}

// Initialize synchronization when the document is ready
document.addEventListener('DOMContentLoaded', function() {
    const serverAddress = localStorage.getItem('serverAddress') || 'http://localhost:5000';
    const urlParams = new URLSearchParams(window.location.search);
    const audioFile = urlParams.get('audio_file');
    const speechMarksFile = urlParams.get('speech_marks_file');

    if (audioFile && speechMarksFile) {
        const audioElement = document.getElementById('audio');
        const audioSource = document.createElement('source');
        audioSource.src = `${serverAddress}${audioFile}`;
        audioSource.type = 'audio/ogg';
        audioElement.appendChild(audioSource);

        initializeSynchronization(`${speechMarksFile}`);
    } else {
        console.error('Audio or speech marks file not provided.');
    }
});
