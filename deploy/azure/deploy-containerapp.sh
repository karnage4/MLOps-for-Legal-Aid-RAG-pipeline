#!/usr/bin/env bash
# Deploy the Legal-Aid RAG API image to Azure Container Apps.
#
# Run in Azure Cloud Shell (Bash) from the repository root:
#   IMAGE=docker.io/<dockerhub-user>/legal-aid-rag-api:<git-sha> \
#   LOCATION=<allowed-region> \
#   bash deploy/azure/deploy-containerapp.sh
#
# First run creates the resource group, Container Apps environment, and app,
# and asks for the API keys (typed input is hidden and never echoed).
# Later runs with a new IMAGE only roll the app to that image (new revision),
# which is also how to roll back: re-run with an older image tag.
# Add UPDATE_KEYS=1 to a later run to replace both API keys as well.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/read-key.sh"

RESOURCE_GROUP="${RESOURCE_GROUP:-legal-aid-rag-rg}"
ENVIRONMENT="${ENVIRONMENT:-legal-aid-rag-env}"
APP_NAME="${APP_NAME:-legal-aid-rag-api}"
LOCATION="${LOCATION:-}"
IMAGE="${IMAGE:-}"

if [[ -z "$IMAGE" ]]; then
  echo "Set IMAGE, e.g. IMAGE=docker.io/<dockerhub-user>/legal-aid-rag-api:<git-sha>" >&2
  exit 1
fi

az extension add --name containerapp --upgrade --only-show-errors

if [[ -z "$LOCATION" ]]; then
  echo "LOCATION is not set. Regions this subscription may deploy to:"
  az policy assignment list --disable-scope-strict-match \
    --query "[].parameters.listOfAllowedLocations.value" -o tsv 2>/dev/null \
    | tr '\t' '\n' | sort -u
  echo "(If the list is empty, check Azure Portal > Policy > Assignments.)"
  echo "Re-run with LOCATION=<one of these regions>." >&2
  exit 1
fi

echo "Registering resource providers (the first run can take a few minutes)..."
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait

az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

if ! az containerapp env show --name "$ENVIRONMENT" \
    --resource-group "$RESOURCE_GROUP" --output none 2>/dev/null; then
  echo "Creating Container Apps environment $ENVIRONMENT in $LOCATION..."
  az containerapp env create --name "$ENVIRONMENT" \
    --resource-group "$RESOURCE_GROUP" --location "$LOCATION" --output none
fi

if az containerapp show --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" --output none 2>/dev/null; then
  update_args=(--image "$IMAGE")
  if [[ "${UPDATE_KEYS:-0}" == 1 ]]; then
    PINECONE_API_KEY="$(read_key PINECONE_API_KEY pcsk_)"
    GROQ_API_KEY="$(read_key GROQ_API_KEY gsk_)"
    echo "Storing the new keys as Container Apps secrets..."
    az containerapp secret set --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" \
      --secrets "pinecone-api-key=$PINECONE_API_KEY" "groq-api-key=$GROQ_API_KEY" \
      --output none
    unset PINECONE_API_KEY GROQ_API_KEY
    # Running replicas keep the old values. A changed variable starts a new
    # revision that reads the new secrets; Express environments do not support
    # "az containerapp revision restart".
    update_args+=(--set-env-vars "KEYS_ROTATED_AT=$(date -u +%Y%m%dT%H%M%SZ)")
  fi
  echo "Updating $APP_NAME to $IMAGE..."
  az containerapp update --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" \
    "${update_args[@]}" --output none
else
  PINECONE_API_KEY="$(read_key PINECONE_API_KEY pcsk_)"
  GROQ_API_KEY="$(read_key GROQ_API_KEY gsk_)"
  # Optional: pull as a Docker Hub user instead of anonymously. Anonymous
  # pulls are limited per IP address, and Azure egress IPs are shared.
  read -r -p "Docker Hub username for pulls (Enter to pull anonymously): " DOCKERHUB_USER
  registry_args=()
  if [[ -n "$DOCKERHUB_USER" ]]; then
    read -r -s -p "Docker Hub read-only access token (hidden): " DOCKERHUB_TOKEN; echo
    registry_args=(--registry-server docker.io
                   --registry-username "$DOCKERHUB_USER"
                   --registry-password "$DOCKERHUB_TOKEN")
  fi

  echo "Creating $APP_NAME (1 vCPU, 2 GiB, scales to zero when idle)..."
  az containerapp create --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" \
    --environment "$ENVIRONMENT" --image "$IMAGE" \
    --target-port 8000 --ingress external \
    --cpu 1.0 --memory 2.0Gi --min-replicas 0 --max-replicas 1 \
    ${registry_args[@]+"${registry_args[@]}"} \
    --secrets "pinecone-api-key=$PINECONE_API_KEY" "groq-api-key=$GROQ_API_KEY" \
    --env-vars \
      PINECONE_API_KEY=secretref:pinecone-api-key \
      GROQ_API_KEY=secretref:groq-api-key \
      GROQ_MODEL=qwen/qwen3.8-27b \
      GROQ_MAX_ATTEMPTS=3 \
      BM25_CANDIDATE_K=10 \
      SEMANTIC_CANDIDATE_K=10 \
      RERANK_CANDIDATE_K=12 \
      PINECONE_QUERY_MAX_ATTEMPTS=3 \
      PINECONE_RETRY_BASE_SECONDS=1.5 \
      PINECONE_INIT_MAX_ATTEMPTS=5 \
      MLFLOW_EXPERIMENT_NAME=legal-aid-rag-azure \
    --output none
  unset PINECONE_API_KEY GROQ_API_KEY DOCKERHUB_TOKEN
fi

FQDN="$(az containerapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn --output tsv)"
URL="https://$FQDN"

echo "Waiting for $URL/health (a cold start downloads the image and loads both retrievers)..."
for _ in $(seq 1 30); do
  if curl --fail --silent --show-error --max-time 30 "$URL/health"; then
    echo
    # Smoke-test the monitoring endpoint too (aggregates only, no user data).
    if ! curl --fail --silent --show-error --max-time 30 "$URL/stats"; then
      echo >&2
      echo "The app is healthy but $URL/stats failed; check the logs." >&2
      exit 1
    fi
    echo
    echo "Live API: $URL"
    echo "Docs:     $URL/docs"
    echo "Stats:    $URL/stats"
    exit 0
  fi
  sleep 10
done

echo "The health check has not passed yet. Inspect the logs with:" >&2
echo "  az containerapp logs show --name $APP_NAME --resource-group $RESOURCE_GROUP --tail 100" >&2
exit 1
