#!/bin/bash

MODEL_URL=${MODEL_URL:-''}
PUBLIC_PORT=${PUBLIC_PORT:-11717}
IMAGE_TAG=${IMAGE_TAG:-'sound-recognizer-app:latest'}
NGROK_TOKEN=${NGROK_TOKEN:-''}
GRADIO_CONTAINER_NAME='gradio'
NGROK_CONTAINER_NAME='ngrok'
NGROK_NETWORK='netgrok'

echo "Starting [ $IMAGE_TAG ] listening on port [ $PUBLIC_PORT ] with model endpoint in [ $MODEL_URL ]"
echo -n "container name: $GRADIO_CONTAINER_NAME. Container ID: "
docker run -it --rm -d -p"$PUBLIC_PORT":11717 --network "$NGROK_NETWORK" --name "$GRADIO_CONTAINER_NAME" "$IMAGE_TAG" --model_url "$MODEL_URL"

echo "Starting ngrok on port [ $PUBLIC_PORT ]"
echo -n "container name: $NGROK_CONTAINER_NAME. Container ID: "
docker run -it --rm -d -p4040:4040 --network "$NGROK_NETWORK" --name "$NGROK_CONTAINER_NAME" -e NGROK_AUTHTOKEN="$NGROK_TOKEN" ngrok/ngrok http "$GRADIO_CONTAINER_NAME":"$PUBLIC_PORT"

./ngrok_status.sh

echo "Extracting public URL"
NGROK_PUBLIC_URL=""
while [ -z "$NGROK_PUBLIC_URL" ]; do
  sleep 1
  # Run 'curl' against ngrok API and extract public (using 'sed' command)
  NGROK_PUBLIC_URL=$(curl --silent --max-time 10 --connect-timeout 5 \
                     http://127.0.0.1:4040/api/tunnels | \
                     sed -nE 's/.*public_url":"https:..([^"]*).*/\1/p')
done
echo "Ngrok public url => [ $NGROK_PUBLIC_URL ]"
echo "$NGROK_PUBLIC_URL" > ngrok-public-url
export NGROK_PUBLIC_URL
