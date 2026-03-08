const axios = require('axios');

class ChatManager {
    constructor(io, systemName = "GridCore") {
        this.io = io;
        this.systemName = systemName;
        this.llmEndpoint = 'http://127.0.0.1:1234/v1/chat/completions';
        this.history = []; // Keep last 50 messages
        
        // HEAVY Blacklist (Regex patterns for safety)
        // Blocks: hate speech, severe profanity, sexual content, spam
        this.blacklist = [
            /nigger/i, /faggot/i, /retard/i, /kys/i, /kill yourself/i,
            /cunt/i, /whore/i, /slut/i, /rape/i, /hitler/i, /nazi/i,
            /chink/i, /tranny/i, /shemale/i, /kkk/i,
            /child porn/i, /cp/i,
            /discord\.gg/i, /t\.me/i // Anti-spam
        ];
    }

    sanitize(text) {
        // 1. Check Blacklist
        for (const pattern of this.blacklist) {
            if (pattern.test(text)) {
                return null; // Message rejected
            }
        }
        // 2. Escape HTML
        return text.replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    async handleMessage(socket, data) {
        const { text, author } = data;
        const cleanText = this.sanitize(text);

        if (!cleanText) {
            // Shadow-ban or warn user? For now, just ignore.
            socket.emit('chat-error', 'Message blocked by Safety Protocol.');
            return;
        }

        const msgObj = {
            id: Date.now(),
            author: author || 'Anonymous',
            text: cleanText,
            role: 'user',
            timestamp: new Date().toLocaleTimeString()
        };

        this.broadcast(msgObj);

        // AI Trigger
        if (cleanText.toLowerCase().includes(`@${this.systemName.toLowerCase()}`) || 
            cleanText.toLowerCase().includes('gridcore')) {
            await this.triggerAI(cleanText, author);
        }
    }

    broadcast(msg) {
        this.history.push(msg);
        if (this.history.length > 50) this.history.shift();
        this.io.emit('chat-msg', msg);
    }

    async triggerAI(userPrompt, user) {
        // Typing indicator
        this.io.emit('chat-typing', { user: this.systemName });

        const systemPrompt = `You are GridCore, an AI overseeing a distributed computing network. 
        Users are donating computing power. Be helpful, technical, slightly futuristic/cyberpunk, but friendly. 
        Keep responses short (under 3 sentences). 
        Current User: ${user}.`;

        try {
            const response = await axios.post(this.llmEndpoint, {
                model: "qwen2.5-vl-7b", // Or local-model
                messages: [
                    { role: "system", content: systemPrompt },
                    { role: "user", content: userPrompt }
                ],
                temperature: 0.7,
                max_tokens: 150
            });

            const replyText = response.data.choices[0].message.content;

            const botMsg = {
                id: Date.now() + 1,
                author: this.systemName,
                text: replyText,
                role: 'system',
                timestamp: new Date().toLocaleTimeString()
            };

            this.broadcast(botMsg);

        } catch (error) {
            console.error("AI Error:", error.message);
            this.broadcast({
                id: Date.now(),
                author: this.systemName,
                text: "[SYSTEM ERROR: Neural Link Offline. Check local LLM backend.]",
                role: 'system',
                isError: true
            });
        }
    }

    sendHistory(socket) {
        socket.emit('chat-history', this.history);
    }
}

module.exports = ChatManager;
