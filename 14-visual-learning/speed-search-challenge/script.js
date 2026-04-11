// The Data - 50 items (1 target, 49 distractors)
const targetItem = "Mirrorless Camera";
const distractors = [
    "Smartphone",
    "Laptop computer",
    "Tablet",
    "Smartwatch",
    "Wireless headphones",
    "Bluetooth speaker",
    "Fitness tracker",
    "E-reader",
    "Gaming console",
    "VR headset",
    "Desktop PC",
    "Computer monitor",
    "Mechanical keyboard",
    "Wireless mouse",
    "External hard drive",
    "USB flash drive",
    "Power bank",
    "Charging cable",
    "Wall adapter",
    "Surge protector",
    "Smart home hub",
    "Smart thermostat",
    "Smart light bulb",
    "Security camera",
    "Video doorbell",
    "Robot vacuum",
    "Coffee maker",
    "Toaster oven",
    "Microwave",
    "Blender",
    "Food processor",
    "Electric kettle",
    "Hair dryer",
    "Electric toothbrush",
    "Electric shaver",
    "Digital thermometer",
    "Blood pressure monitor",
    "Smart scale",
    "Drone",
    "Action camera",
    "Tripod",
    "Microphone",
    "Webcam",
    "Ring light",
    "Projector",
    "Printer",
    "Scanner",
    "Shredder",
    "Laminator"
];

// Combine and shuffle
function shuffle(array) {
    let currentIndex = array.length, randomIndex;
    while (currentIndex != 0) {
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;
        [array[currentIndex], array[randomIndex]] = [array[randomIndex], array[currentIndex]];
    }
    return array;
}

let allVariants = [...distractors, targetItem];
allVariants = shuffle(allVariants);

// App State
let r1StartTime = 0;
let r1Interval = null;
let r1Elapsed = 0;

let r2StartTime = 0;
let r2Interval = null;
let r2Elapsed = 0;

// DOM Elements
const views = document.querySelectorAll('.view');
const btnStartRound1 = document.getElementById('btn-start-round1');
const gridR1 = document.getElementById('card-grid-r1');
const timerR1 = document.getElementById('timer-r1');

const btnRunVector = document.getElementById('btn-run-vector');
const gridR2 = document.getElementById('card-grid-r2');
const timerR2 = document.getElementById('timer-r2');

const finalTimeR1 = document.getElementById('final-time-r1');
const finalTimeR2 = document.getElementById('final-time-r2');
const btnRestart = document.getElementById('btn-restart');

// View Control
function showView(viewId) {
    views.forEach(v => v.classList.remove('active'));
    document.getElementById(viewId).classList.add('active');
    window.scrollTo(0, 0);
}

// Timer Format (Returns "X.XX" format)
function formatTime(ms) {
    return (ms / 1000).toFixed(2);
}

// ------ Round 1 Logic ------

function initRound1() {
    gridR1.innerHTML = '';

    // Create grid items
    allVariants.forEach((text, index) => {
        const card = document.createElement('div');
        card.className = 'search-card';
        card.textContent = text;
        card.dataset.isTarget = text === targetItem;

        card.addEventListener('click', () => handleR1Click(card));
        gridR1.appendChild(card);
    });

    // Start Timer
    document.querySelector('#view-round1 .timer-display').classList.add('running');
    r1StartTime = Date.now();
    r1Interval = setInterval(() => {
        r1Elapsed = Date.now() - r1StartTime;
        timerR1.textContent = formatTime(r1Elapsed);
    }, 10); // Update every 10ms for smooth counter
}

function handleR1Click(cardDiv) {
    if (!r1Interval) return; // ignore clicks if stopped

    const isTarget = cardDiv.dataset.isTarget === 'true';

    if (isTarget) {
        // Success
        clearInterval(r1Interval);
        r1Interval = null; // marks it stopped

        cardDiv.classList.add('correct');
        document.querySelector('#view-round1 .timer-display').classList.remove('running');
        document.querySelector('#view-round1 .timer-display').classList.add('stopped');

        // Wait briefly, then move to round 2
        setTimeout(() => {
            showView('view-round2');
            initRound2();
        }, 1500);

    } else {
        // Incorrect
        cardDiv.classList.add('incorrect');
        setTimeout(() => {
            cardDiv.classList.remove('incorrect');
        }, 400); // Remove animation class so it can shake again
    }
}

// ------ Round 2 Logic ------

function initRound2() {
    gridR2.innerHTML = '';
    timerR2.textContent = "0.00";
    document.querySelector('#view-round2 .timer-display').classList.remove('stopped');
    document.querySelector('#view-round2 .timer-display').classList.remove('running');

    // Create grid items
    allVariants.forEach((text) => {
        const card = document.createElement('div');
        card.className = 'search-card';
        card.textContent = text;
        card.dataset.isTarget = text === targetItem;
        gridR2.appendChild(card);
    });
}

function runVectorSearch() {
    // Prevent double clicking
    btnRunVector.disabled = true;

    // Start fast timer simulation
    document.querySelector('#view-round2 .timer-display').classList.add('running');
    r2StartTime = Date.now();

    // Simulate "instant" search that still looks like it's calculating
    r2Interval = setInterval(() => {
        const elapsed = Date.now() - r2StartTime;
        timerR2.textContent = formatTime(elapsed);

        // Stop after ~0.05 - 0.15s randomly for realism
        if (elapsed > (40 + Math.random() * 80)) {
            clearInterval(r2Interval);
            finishVectorSearch(elapsed);
        }
    }, 10);
}

function finishVectorSearch(elapsed) {
    r2Elapsed = elapsed;
    timerR2.textContent = formatTime(r2Elapsed);

    document.querySelector('#view-round2 .timer-display').classList.remove('running');
    document.querySelector('#view-round2 .timer-display').classList.add('stopped');

    // Find target and highlight
    const cards = gridR2.querySelectorAll('.search-card');
    cards.forEach(card => {
        if (card.dataset.isTarget === 'true') {
            card.classList.add('highlighted');
            // Scroll to it if needed
            card.scrollIntoView({ behavior: 'smooth', block: 'center' });
        } else {
            // Dim others
            card.style.opacity = '0.3';
        }
    });

    // Proceed to Conclusion
    setTimeout(() => {
        showConclusion();
    }, 2000);
}

// ------ Conclusion Logic ------

function showConclusion() {
    showView('view-conclusion');

    finalTimeR1.textContent = formatTime(r1Elapsed);

    // Animate the R2 timer dramatically
    let displayTime = 0;
    const targetTime = r2Elapsed;
    const duration = 1000; // 1 second dramatic roll
    const steps = 30;
    const increment = targetTime / steps;
    const stepTime = duration / steps;

    const countInterval = setInterval(() => {
        displayTime += increment;
        if (displayTime >= targetTime) {
            displayTime = targetTime;
            clearInterval(countInterval);

            // Add extra pop when finished
            finalTimeR2.parentElement.style.transform = 'scale(1.1)';
            setTimeout(() => {
                finalTimeR2.parentElement.style.transform = 'scale(1)';
                finalTimeR2.parentElement.style.transition = 'transform 0.3s ease';
            }, 300);
        }
        finalTimeR2.textContent = formatTime(displayTime);
    }, stepTime);
}

// ------ Event Listeners ------

btnStartRound1.addEventListener('click', () => {
    showView('view-round1');
    initRound1();
});

btnRunVector.addEventListener('click', runVectorSearch);

btnRestart.addEventListener('click', () => {
    // Reshuffle for replayability
    allVariants = shuffle([...distractors, targetItem]);
    btnRunVector.disabled = false;
    showView('view-intro');
});

// Start on Intro
showView('view-intro');
