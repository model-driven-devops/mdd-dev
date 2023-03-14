#!/usr/bin/env bash

IMAGE=ghcr.io/model-driven-devops/mdd:latest

OPTIONS="--env ANSIBLE_PYTHON_INTERPRETER=/usr/bin/python3"
OPTIONS="$OPTIONS --env COLLECTIONS_PATHS=/"
if [[ ! -z "$ANSIBLE_VAULT_PASSWORD_FILE" ]]; then
   OPTIONS="--env ANSIBLE_VAULT_PASSWORD_FILE=/tmp/vault.pw -v $ANSIBLE_VAULT_PASSWORD_FILE:/tmp/vault.pw"
fi

OPTION_LIST=( \
   "CML_HOST" \
   "CML_USERNAME" \
   "CML_PASSWORD" \
   "CML_LAB" \
   "CML_VERIFY_CERT" \
   "NSO_URL" \
   "NSO_USERNAME" \
   "NSO_PASSWORD" \
   "ANSIBLE_INVENTORY" \
   "ANSIBLE_COLLECTIONS_PATH" \
   )

for OPTION in ${OPTION_LIST[*]}; do
   if [[ ! -z "${!OPTION}" ]]; then
      OPTIONS="$OPTIONS --env $OPTION=${!OPTION}"
   fi
done

while getopts ":dl" opt; do
  case $opt in
    d)
      docker run -it --rm -v $PWD:/ansible --env PWD="/ansible" --env USER="$USER" $OPTIONS $IMAGE /bin/bash
      exit
      ;;
    l)
      docker run -it --rm -v $PWD:/ansible --env PWD="/ansible" --env USER="$USER" $OPTIONS $IMAGE ansible-lint
      exit
      ;;
  esac
done
docker run -it --rm -v $PWD:/ansible --env PWD="/ansible" --env USER="$USER" $OPTIONS $IMAGE ansible-playbook "$@"
