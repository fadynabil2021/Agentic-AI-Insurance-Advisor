FROM node:20-alpine

WORKDIR /app

# Ensure we have the right permissions
RUN chown node:node /app
USER node

# Copy package files
COPY --chown=node:node package.json package-lock.json* ./

# Install dependencies (ignoring scripts to avoid issues)
RUN npm install --legacy-peer-deps

# Build the production application
RUN npm run build

# Start production server
CMD ["sh", "-c", "npm run start -- -p ${PORT:-3000}"]
