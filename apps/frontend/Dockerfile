# NovaCare Frontend - Dockerfile

FROM node:18-slim

WORKDIR /app

# Install dependencies first for better caching
COPY package.json package-lock.json* ./
RUN npm install

# Copy the rest of the application
COPY . .

# Build the application
RUN npm run build

# Expose the frontend port
EXPOSE 3000

# Start in production mode
CMD ["npm", "start"]
