// modules/hashing.js
const crypto = require('crypto');

module.exports = {
    execute: async (payload) => {
        const { targetHash, charset, length, prefix } = payload;
        
        // Brute force all strings starting with 'prefix' up to 'length'
        // This is a simplified version that just checks specific length
        // A real cracker would check 1..length
        
        const result = { found: false, secret: null };

        function generate(current) {
            if (result.found) return;
            
            if (current.length === length) {
                const hash = crypto.createHash('md5').update(current).digest('hex');
                if (hash === targetHash) {
                    result.found = true;
                    result.secret = current;
                }
                return;
            }

            for (let i = 0; i < charset.length; i++) {
                generate(current + charset[i]);
            }
        }

        // Start generation with the assigned prefix
        generate(prefix);
        
        return result;
    }
};
