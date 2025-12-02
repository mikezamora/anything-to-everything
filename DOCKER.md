docker build --cache-from=type=registry,ref=localhost:5065/anything-to-everything:cache --cache-to=type=registry,ref=localhost:5065/anything-to-everything:cache,mode=max -t anything-to-everything .

docker compose down
docker compose up -d
docker compose logs -f