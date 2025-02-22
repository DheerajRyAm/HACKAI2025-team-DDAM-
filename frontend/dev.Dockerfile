FROM node:20-alpine AS base

FROM base AS builder
RUN apk add --no-cache libc6-compat
WORKDIR /app

COPY package.json package-lock.json ./
RUN if [ -f package-lock.json ]; then npm ci; \
    else echo "Lockfile not found" && exit 1; \
    fi

COPY . .

CMD ["npm", "run", "dev"]