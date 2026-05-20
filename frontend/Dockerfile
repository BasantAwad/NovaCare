# Use a lightweight Node.js 18 image (bullseye-slim is highly compatible with ARM64 edge devices)
FROM node:18-bullseye-slim

WORKDIR /app

# Copy dependency specifications
COPY package*.json ./

# Install dependencies cleanly
RUN npm install --legacy-peer-deps --no-audit --no-fund

# Copy all application code (including .env.local if present)
COPY . .

# Build Next.js for production
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# Expose standard port
EXPOSE 3000

# Start Next.js server
CMD ["npm", "run", "start"]
