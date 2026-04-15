FROM node:20-alpine

WORKDIR /app

# Ensure we have the right permissions
RUN chown node:node /app
USER node

# Copy package files
COPY --chown=node:node package.json package-lock.json* ./

# Install dependencies (ignoring scripts to avoid issues)
RUN npm install --legacy-peer-deps

# Copy source code
COPY --chown=node:node . .

# Expose port
EXPOSE 3000

# Start development server binding to all interfaces and custom port
CMD sh -c "npm run dev -- -H 0.0.0.0 -p ${PORT:-3000}"
