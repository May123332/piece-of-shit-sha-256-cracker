const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const cors = require('cors');
const path = require('path');
const localtunnel = require('localtunnel');
const ChatManager = require('./chat');

const app = express();
app.use(cors());
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

const server = http.createServer(app);
const io = new Server(server, { cors: { origin: "*" } });

// --- SYSTEMS ---
const chat = new ChatManager(io);

// --- STATE ---
let workers = new Map(); 
let jobQueue = [];       
let activeJobs = new Map();
let globalStats = { totalOps: 0, connectedCores: 0 };
let publicUrl = "http://localhost:3000";
let masterKey = "admin"; // Default simple key

// --- JOB MANAGERS ---
const JobSplitters = {
    'fractal': (payload) => {
        const chunks = [];
        const { width, height, jobId } = payload;
        const TILE_SIZE = 100;
        for (let y = 0; y < height; y += TILE_SIZE) {
            for (let x = 0; x < width; x += TILE_SIZE) {
                chunks.push({
                    jobId, type: 'fractal',
                    payload: { x, y, width: Math.min(TILE_SIZE, width - x), height: Math.min(TILE_SIZE, height - y), fullWidth: width, fullHeight: height, maxIter: payload.maxIter || 100 }
                });
            }
        }
        return chunks;
    },
        'hash-crack': (payload) => {
            const chunks = [];
            const { targetHash, charset, length, jobId, algorithm } = payload;
            for (let i = 0; i < charset.length; i++) {
                chunks.push({
                    jobId,
                    type: 'hash-crack',
                    payload: {
                        targetHash,
                        charset,
                        length,
                        prefix: charset[i],
                        algorithm: algorithm || 'md5'
                    }
                });
            }
            return chunks;
        },
    'llm-prompt': (payload) => [{ jobId: payload.jobId, type: 'llm-prompt', payload: payload }],
    'minecraft-seed': (payload) => {
        const chunks = [];
        const { rangeStart, rangeEnd, jobId } = payload;
        const step = 1000000;
        for (let i = rangeStart; i < rangeEnd; i += step) {
            chunks.push({ jobId, type: 'minecraft-seed', payload: { start: i, end: Math.min(i + step, rangeEnd), targetBiome: payload.targetBiome } });
        }
        return chunks;
    },
    'server-check': (payload) => {
        const chunks = [];
        const { ips, jobId } = payload;
        const CHUNK_SIZE = 50; 
        for (let i = 0; i < ips.length; i += CHUNK_SIZE) {
            chunks.push({
                jobId,
                type: 'server-check',
                payload: { ips: ips.slice(i, i + CHUNK_SIZE) }
            });
        }
        return chunks;
    }
};

// --- API ENDPOINTS ---

app.get('/join', (req, res) => {
    // Generate a bash script that downloads the correct binary
    const script = `
#!/bin/bash
SERVER="${publicUrl}"
OS="$(uname -s)"
ARCH="$(uname -m)"

echo ">> GRIDOS GHOST AGENT INITIALIZING..."
echo ">> DETECTING OS: $OS $ARCH"

if [ "$OS" == "Linux" ]; then
    URL="$SERVER/bin/grid-compute-linux"
elif [ "$OS" == "Darwin" ]; then
    URL="$SERVER/bin/grid-compute-macos"
else
    echo ">> UNSUPPORTED OS. USE WINDOWS LAUNCHER."
    exit 1
fi

DEST="/tmp/ghost-agent"
echo ">> DOWNLOADING PAYLOAD..."
curl -sL "$URL" -o "$DEST"
chmod +x "$DEST"

echo ">> EXECUTING GHOST PROTOCOL..."
"$DEST" "$SERVER" "Ghost-$(whoami)"

echo ">> CLEANING UP TRACES..."
rm "$DEST"
    `;
    res.setHeader('Content-Type', 'text/plain');
    res.send(script);
});

app.get('/api/config', (req, res) => {
    res.json({ url: publicUrl, localUrl: `http://${getLocalIp()}:3000` });
});

// Admin-only Job Submission
app.post('/api/job', (req, res) => {
    // In a real app, check req.headers.authorization === masterKey
    const { type, payload } = req.body;
    const jobId = Date.now().toString();

    if (!JobSplitters[type]) return res.status(400).json({ error: 'Unknown job type' });

    const chunks = JobSplitters[type]({ ...payload, jobId });
    activeJobs.set(jobId, { type, startTime: Date.now(), totalChunks: chunks.length, completedChunks: 0, results: [] });
    jobQueue.push(...chunks);
    
    io.emit('job-update', getJobSummary());
    processQueue();
    res.json({ success: true, jobId, chunks: chunks.length });
});

function getJobSummary() {
    return Array.from(activeJobs.entries()).map(([id, job]) => ({
        id, type: job.type,
        progress: Math.round((job.completedChunks / job.totalChunks) * 100) || 0,
        results: job.results.length 
    }));
}

function getLocalIp() {
    const { networkInterfaces } = require('os');
    const nets = networkInterfaces();
    for (const name of Object.keys(nets)) {
        for (const net of nets[name]) {
            if (net.family === 'IPv4' && !net.internal) return net.address;
        }
    }
    return 'localhost';
}

function processQueue() {
    if (jobQueue.length === 0) return;
    for (const [socketId, worker] of workers) {
        if (!worker.currentTask && jobQueue.length > 0) {
            // Logic to skip LLM tasks for non-LLM nodes
            const taskIndex = jobQueue.findIndex(t => t.type !== 'llm-prompt' || worker.specs.hasLLM);
            
            if (taskIndex !== -1) {
                const task = jobQueue.splice(taskIndex, 1)[0];
                worker.currentTask = task.jobId;
                workers.set(socketId, worker);
                io.to(socketId).emit('task', task);
                io.emit('worker-update', Array.from(workers.values()));
            }
        }
    }
}

// --- SOCKETS ---

io.on('connection', (socket) => {
    // 1. CHAT
    chat.sendHistory(socket);
    socket.on('chat-send', (data) => chat.handleMessage(socket, data));

    // 2. WORKER REGISTRATION
    socket.on('register', (specs) => {
        workers.set(socket.id, {
            id: socket.id,
            specs: specs || {},
            currentTask: null,
            name: specs.name || 'Volunteer'
        });
        globalStats.connectedCores += (specs.cores || 1);
        io.emit('worker-update', Array.from(workers.values()));
        processQueue();
    });

    // 3. TASK RESULTS
    socket.on('result', (result) => {
        const worker = workers.get(socket.id);
        if (worker) {
            worker.currentTask = null;
            workers.set(socket.id, worker);
        }

        const job = activeJobs.get(result.jobId);
        if (job) {
            job.completedChunks++;
            job.results.push(result.data);
            io.emit('job-result', { 
                jobId: result.jobId, type: job.type, data: result.data, 
                progress: job.completedChunks / job.totalChunks
            });
        }
        processQueue();
        io.emit('worker-update', Array.from(workers.values()));
    });

    socket.on('disconnect', () => {
        const worker = workers.get(socket.id);
        if (worker) {
            globalStats.connectedCores -= (worker.specs.cores || 0);
            workers.delete(socket.id);
            io.emit('worker-update', Array.from(workers.values()));
        }
    });
});

const PORT = 3000;
server.listen(PORT, '0.0.0.0', async () => {
    console.log(`GridOS Core running on ${PORT}`);
    try {
        const tunnel = await localtunnel({ port: PORT });
        publicUrl = tunnel.url;
        console.log(`Public URL: ${publicUrl}`);
    } catch (e) { console.error("Tunnel Error:", e); }
});
