FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY index.html vite.config.js ./
COPY src ./src
# The container serves the graph TOP-15 from /api/search, never the mock.
ARG VITE_MOCK=0
ENV VITE_MOCK=$VITE_MOCK
# 1 = the light search-only build (compose.search.yaml): no ingestion view.
ARG VITE_SEARCH_ONLY=0
ENV VITE_SEARCH_ONLY=$VITE_SEARCH_ONLY
RUN npm run build

FROM nginx:alpine
# The public domain (LCTREND_DOMAIN) is filled in when the container starts.
ENV LCTREND_DOMAIN="" \
    NGINX_ENVSUBST_FILTER=^LCTREND_
COPY nginx.conf /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
