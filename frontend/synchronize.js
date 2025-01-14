function preprocessWordTimesExact(fragments) {
    const lookupTable = [];

    fragments.forEach((fragment) => {
        const { begin, end, text, bounding_box, page_number } = fragment;
        lookupTable.push({
            start: parseFloat(begin) * 1000, // Convert to milliseconds
            end: parseFloat(end) * 1000,     // Convert to milliseconds
            text,
            bounding_box,
            page_number,
        });
    });

    return lookupTable;
}

function initializeSynchronization(speechMarksFile) {
    const serverAddress = localStorage.getItem("serverAddress") || "http://localhost:5000";
    const audioElement = document.getElementById("audio");
    const playbackRateSelector = document.getElementById("playback-rate");
    const skipBackwardButton = document.getElementById("skip-backward");
    const skipForwardButton = document.getElementById("skip-forward");

    const audio = new MediaElementPlayer(audioElement, {
        success: function (mediaElement, originalNode) {
            mediaElement.addEventListener("timeupdate", syncText);
            mediaElement.addEventListener("ratechange", resetText);
            mediaElement.addEventListener("seeked", handleSeek);
        },
    });

    let lookupTable = [];
    let currentFragmentIndex = 0;
    let currentHighlight = null;
    let highlightDiv = null; // Single highlight div
    let debounceScrollTimeout = null;

    fetch(`${serverAddress}${speechMarksFile}`)
        .then((response) => response.json())
        .then((data) => {
            lookupTable = preprocessWordTimesExact(data.fragments);
            renderWordsForCurrentTime(audio.media.currentTime * 1000);
        })
        .catch((error) => {
            console.error("Error loading word times:", error);
        });

    function syncText() {
        const currentTime = audio.media.currentTime * 1000;
        renderWordsForCurrentTime(currentTime);
    }

    function resetText() {
        renderWordsForCurrentTime(audio.media.currentTime * 1000);
    }

    function handleSeek() {
        renderWordsForCurrentTime(audio.media.currentTime * 1000);
    }

    function renderWordsForCurrentTime(currentTime) {
        let fragment = null;

        if (
            lookupTable[currentFragmentIndex] &&
            currentTime >= lookupTable[currentFragmentIndex].start &&
            currentTime <= lookupTable[currentFragmentIndex].end
        ) {
            fragment = lookupTable[currentFragmentIndex];
        } else {
            for (let i = 0; i < lookupTable.length; i++) {
                if (
                    currentTime >= lookupTable[i].start &&
                    currentTime <= lookupTable[i].end
                ) {
                    fragment = lookupTable[i];
                    currentFragmentIndex = i;
                    break;
                }
            }
        }

        if (fragment) {
            highlightBoundingBox(fragment);
        }
    }

    function highlightBoundingBox(fragment) {
        const pageNumber = fragment.page_number;
        const boundingBox = fragment.bounding_box;
    
        if (!pageNumber || !boundingBox) return;
    
        const pageIndex = pageNumber - 1;
        const pageInfo = pdfPages[pageIndex];
    
        if (!pageInfo) return;
    
        const { containerDiv, viewport } = pageInfo;
        const scale = viewport.scale;
    
        const [x0, top, x1, bottom] = boundingBox;
    
        // If we haven't created the highlightDiv yet, create it.
        if (!highlightDiv) {
            highlightDiv = document.createElement("div");
            highlightDiv.classList.add("pdf-highlight");
            containerDiv.appendChild(highlightDiv);
        } else if (highlightDiv.parentNode !== containerDiv) {
            // If the highlight is currently attached to a different page's container, remove it first.
            highlightDiv.parentNode.removeChild(highlightDiv);
            containerDiv.appendChild(highlightDiv);
        }
    
        // Update the position and size with scaling
        highlightDiv.style.left = x0 * scale + "px";
        highlightDiv.style.top = top * scale + "px";
        highlightDiv.style.width = (x1 - x0) * scale + "px";
        highlightDiv.style.height = (bottom - top) * scale + "px";
    
        // Smoothly scroll to the highlighted area using debouncing
        if (debounceScrollTimeout) {
            clearTimeout(debounceScrollTimeout);
        }
        debounceScrollTimeout = setTimeout(() => {
            highlightDiv.scrollIntoView({
                behavior: "smooth",
                block: "center",
                inline: "nearest",
            });
        }, 100); // Adjust delay as needed
    }
    

    playbackRateSelector.addEventListener("change", function () {
        audio.media.playbackRate = parseFloat(this.value);
    });

    skipBackwardButton.addEventListener("click", function () {
        audio.media.setCurrentTime(Math.max(0, audio.media.currentTime - 5));
        handleSeek();
    });

    skipForwardButton.addEventListener("click", function () {
        audio.media.setCurrentTime(
            Math.min(audio.media.duration, audio.media.currentTime + 5)
        );
        handleSeek();
    });
}

document.addEventListener("keydown", function (event) {
    // Prevent default actions for specific key combinations
    if (
        (event.key === " " && !event.target.matches('input, textarea')) ||
        (event.ctrlKey && (event.key === "ArrowRight" || event.key === "ArrowLeft"))
    ) {
        event.preventDefault();
    }

    // Spacebar for play/pause toggle
    if (event.key === " " && !event.ctrlKey && !event.shiftKey && !event.altKey && !event.metaKey) {
        if (audio.media.paused) {
            audio.media.play();
        } else {
            audio.media.pause();
        }
    }

    // Ctrl + Right Arrow to skip forward 5 seconds
    if (event.ctrlKey && event.key === "ArrowRight") {
        audio.media.setCurrentTime(
            Math.min(audio.media.duration, audio.media.currentTime + 5)
        );
        handleSeek();
    }

    // Ctrl + Left Arrow to skip backward 5 seconds
    if (event.ctrlKey && event.key === "ArrowLeft") {
        audio.media.setCurrentTime(Math.max(0, audio.media.currentTime - 5));
        handleSeek();
    }
});

// Global array to store references for each PDF page
const pdfPages = [];

document.addEventListener("DOMContentLoaded", function () {
    const serverAddress = localStorage.getItem("serverAddress") || "http://localhost:5000";
    const urlParams = new URLSearchParams(window.location.search);
    const audioFile = urlParams.get("audio_file");
    const speechMarksFile = urlParams.get("speech_marks_file");
    const paperId = urlParams.get("paper_id");

    if (audioFile && speechMarksFile) {
        const audioElement = document.getElementById("audio");
        const audioSource = document.createElement("source");
        audioSource.src = `${serverAddress}${audioFile}`;
        audioSource.type = "audio/ogg";
        audioElement.appendChild(audioSource);

        initializeSynchronization(`${speechMarksFile}`);
    } else {
        console.error("Audio or speech marks file not provided.");
    }

    if (paperId) {
        const pdfContainer = document.getElementById("pdf-render");
        const url = `${serverAddress}/papers/${paperId}/pdf`;

        const renderPDF = async (pdfUrl) => {
            const pdfjsLib = window["pdfjs-dist/build/pdf"];
            pdfjsLib.GlobalWorkerOptions.workerSrc =
                "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.10.377/pdf.worker.min.js";

            const pdf = await pdfjsLib.getDocument(pdfUrl).promise;
            for (let i = 1; i <= pdf.numPages; i++) {
                const page = await pdf.getPage(i);
                const scale = 4;
                const viewport = page.getViewport({ scale });

                const pageContainerDiv = document.createElement("div");
                pageContainerDiv.classList.add("pdf-page-container");
                pageContainerDiv.style.position = "relative"; // Ensure positioning context
                pageContainerDiv.style.width = viewport.width + "px";
                pageContainerDiv.style.height = viewport.height + "px";
                pageContainerDiv.style.margin = "10px auto"; // Center the page

                const canvas = document.createElement("canvas");
                canvas.width = viewport.width;
                canvas.height = viewport.height;
                const context = canvas.getContext("2d");

                const renderContext = {
                    canvasContext: context,
                    viewport: viewport,
                };
                await page.render(renderContext).promise;

                pageContainerDiv.appendChild(canvas);
                pdfContainer.appendChild(pageContainerDiv);

                pdfPages.push({
                    containerDiv: pageContainerDiv,
                    viewport,
                });
            }
        };

        renderPDF(url).catch((error) =>
            console.error("Error rendering PDF:", error)
        );
    } else {
        console.error("Paper ID not provided.");
    }
});
