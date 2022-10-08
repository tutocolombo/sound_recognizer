#!/bin/bash

MODEL_URL=${MODEL_URL:-''}
PUBLIC_PORT=${PUBLIC_PORT:-11717}
CONTAINER_TAG=${CONTAINER_TAG:-'sound-recognizer-app:latest'}

echo "Starting docker container [ $CONTAINER_TAG ] with model endpoint in [ $MODEL_URL ]"
echo -n "Container ID: "
docker run -it --rm -d -p"$PUBLIC_PORT":11717 "$CONTAINER_TAG" --model_url "$MODEL_URL"

echo -n "Start ngrok on port [ $PUBLIC_PORT ] ."
nohup ngrok http "$PUBLIC_PORT" &> ngrok.log &

NGROK_PUBLIC_URL=""
while [ -z "$NGROK_PUBLIC_URL" ]; do
  sleep 1
  echo -n "."
  # Run 'curl' against ngrok API and extract public (using 'sed' command)
  NGROK_PUBLIC_URL=$(curl --silent --max-time 10 --connect-timeout 5 \
                            http://127.0.0.1:4040/api/tunnels | \
                            sed -nE 's/.*public_url":"https:..([^"]*).*/\1/p')
done
echo ". Ngrok public url => [ $NGROK_PUBLIC_URL ]"
echo "$NGROK_PUBLIC_URL" > ngrok-public-url
export NGROK_PUBLIC_URL
