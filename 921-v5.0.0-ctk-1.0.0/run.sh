
#!/bin/bash
# -----------------------------------------------------------------------------
#  Copyright (c) TM Forum. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -----------------------------------------------------------------------------

cd "$(dirname "$0")"

# clean up
rm -rf ./reports
mkdir -p ./reports
rm -f ./REPORT.HTML
 
[ $# -gt 0 ] && export TEST_BASE_URL=$1

docker network inspect tmf >/dev/null 2>&1 || docker network create tmf

export DOCKER_UID=$(id -u) DOCKER_GID=$(id -g)

# determine platform to use when replacing placeholder in compose file
if [ -z "${platform:-}" ]; then
    echo "Error: 'platform' environment variable is not set. Please set platform to a value such as linux/arm64 or linux/amd64, e.g. platform=linux/amd64"
    exit 1
fi
platform_value=$platform

tmp_compose="$(mktemp)"
awk -v plat="$platform_value" '
  /^[[:space:]]*platform:/ && !done {
    sub(/:[[:space:]]*.*/, ": " plat)
    done=1
  }
  { print }
' docker-compose.yaml >"$tmp_compose"

mv $tmp_compose ./docker-compose.yaml

# run docker compose
ctk_container=$(docker compose run -d ctk)
docker logs -f "$ctk_container"


# extract the report
reportPath="./reports/index.html"
modifiedReportPath="./REPORT.HTML"

# Check if the report exists
if [ -f "$reportPath" ]; then
    echo "Modifying the report file..."

    # Modify the report content using sed
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sed -i 's/.state={expanded:!1}/.state={expanded:1}/g' "$reportPath"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' 's/.state={expanded:!1}/.state={expanded:1}/g' "$reportPath"
    else
        echo "Unsupported OS. Cannot modify the report automatically."
        exit 1
    fi

    echo "Copying modified report to $modifiedReportPath..."
    cp $reportPath $modifiedReportPath
else
    echo "Report file $reportPath does not exist. Exiting."
    exit 1
fi

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open $modifiedReportPath
elif [[ "$OSTYPE" == "darwin"* ]]; then
    open $modifiedReportPath
else
    echo "Unsupported OS. Please open REPORT.HTML manually."
fi

cleanup() {
    if [ -n "${ctk_container:-}" ]; then
        docker rm -f "$ctk_container" >/dev/null 2>&1 || true
        
    rm -f "$tmp_compose"
}
trap cleanup EXIT