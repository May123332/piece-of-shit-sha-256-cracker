const io = require('socket.io-client');
const os = require('os');
const { exec } = require('child_process');

// Configuration
const SERVER_URL = process.argv[2] || 'http://localhost:3000';
const WORKER_NAME = process.argv[3] || os.hostname();

console.log(`Connecting to GridCompute Orchestrator at ${SERVER_URL}...`);

const socket = io(SERVER_URL);

// --- CAPABILITY DISCOVERY ---
const specs = {
    name: WORKER_NAME,
    cores: os.cpus().length,
    platform: os.platform(),
    hasLLM: false,
    llmEndpoint: null
};

// Helper to check a URL
const checkService = (url) => {
    return new Promise((resolve) => {
        const req = require('http').get(url, (res) => {
            resolve(res.statusCode === 200);
        }).on('error', () => resolve(false));
    });
};

async function detectCapabilities() {
    // Check Ollama
    if (await checkService('http://localhost:11434/api/tags')) {
        console.log('[Capability] Ollama detected (Port 11434)');
        specs.hasLLM = true;
        specs.llmEndpoint = 'http://localhost:11434';
        specs.llmType = 'ollama';
    } 
    // Check LM Studio / OpenAI Compat (Standard port 1234)
    else if (await checkService('http://localhost:1234/v1/models')) {
        console.log('[Capability] LM Studio/OpenAI Compat detected (Port 1234)');
        specs.hasLLM = true;
        specs.llmEndpoint = 'http://localhost:1234/v1';
        specs.llmType = 'openai';
    }

    socket.emit('register', specs);
}

detectCapabilities();


// --- MODULE LOADER ---
const modules = {
    'fractal': require('./modules/fractal'),
    'hash-crack': require('./modules/hashing'),
    'llm-prompt': require('./modules/llm')
};

socket.on('connect', () => {
    console.log('Connected to Orchestrator!');
    // Registration happens after capability check above, or fallback:
    if (!specs.hasLLM) socket.emit('register', specs);
});

socket.on('task', async (task) => {
    console.log(`[Task] Received ${task.type} (Job ${task.jobId})`);
    
    if (modules[task.type]) {
        try {
            const start = Date.now();
            const resultData = await modules[task.type].execute(task.payload, specs);
            const duration = Date.now() - start;

            console.log(`[Task] Completed ${task.type} in ${duration}ms`);
            
            socket.emit('result', {
                jobId: task.jobId,
                data: resultData
            });
        } catch (e) {
            console.error(`[Error] Task failed:`, e);
        }
    } else {
        console.error(`[Error] Unknown task type: ${task.type}`);
    }
});

socket.on('disconnect', () => {
    console.log('Disconnected from server.');
});
