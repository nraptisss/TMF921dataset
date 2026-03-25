@echo off
REM -----------------------------------------------------------------------------
REM  Copyright (c) TM Forum. All rights reserved.
REM
REM  Licensed under the Apache License, Version 2.0 (the "License");
REM  you may not use this file except in compliance with the License.
REM  You may obtain a copy of the License at
REM
REM      http://www.apache.org/licenses/LICENSE-2.0
REM
REM  Unless required by applicable law or agreed to in writing, software
REM  distributed under the License is distributed on an "AS IS" BASIS,
REM  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
REM  See the License for the specific language governing permissions and
REM  limitations under the License.
REM -----------------------------------------------------------------------------

SETLOCAL EnableExtensions EnableDelayedExpansion

REM ── Work in this script's folder
cd /d "%~dp0"

REM ── Constants
SET "SRC_COMPOSE=docker-compose.yaml"

REM ── Sanity checks
IF NOT EXIST "%SRC_COMPOSE%" (
  echo Error: "%SRC_COMPOSE%" not found in %CD%
  ENDLOCAL & EXIT /b 1
)
IF NOT EXIST "config.json" (
  echo Error: "config.json" not found in %CD%
  echo Ensure the file exists or update the volume path in docker-compose.yaml
  ENDLOCAL & EXIT /b 1
)

REM ── Clean report artifacts
IF EXIST "reports" rmdir /s /q "reports"
mkdir "reports" >nul 2>&1
IF EXIST "REPORT.HTML" del "REPORT.HTML"

REM ── Optional base URL argument
IF NOT "%~1"=="" SET "TEST_BASE_URL=%~1"

REM ── Pick docker compose command (v2 or legacy)
SET "DC=docker compose"
%DC% version >nul 2>&1
IF ERRORLEVEL 1 (
  SET "DC=docker-compose"
  %DC% version >nul 2>&1 || (
    echo Error: docker compose / docker-compose not found.
    ENDLOCAL & EXIT /b 1
  )
)

REM ── Ensure docker network exists
docker network inspect tmf >nul 2>&1 || docker network create tmf >nul 2>&1

REM UID/GID are not relevant on Windows, setting dummy values
set DOCKER_UID=1000
set DOCKER_GID=1000

REM ── Require platform override
IF NOT DEFINED platform (
  echo Error: platform environment variable is not set. Example: platform=linux/arm64 run-ctk.bat
  ENDLOCAL & EXIT /b 1
)

REM ── Create temp compose in this directory with platform substituted
SET "TMP_COMPOSE="
FOR /F "usebackq delims=" %%i IN (`powershell -NoProfile -Command ^
  "$plat = $env:platform;" ^
  "$src = 'docker-compose.yaml';" ^
  "$dst = Join-Path (Get-Location) ('tmf_ctk_compose_' + [guid]::NewGuid().ToString() + '.yaml');" ^
  "$done = $false;" ^
  "$content = Get-Content -Path $src;" ^
  "$content = $content | ForEach-Object { if (-not $done -and $_ -match '^[\s]*platform:') { $done = $true; return ([regex]::Replace($_, '(^\s*platform:\s*).+$', '${1}' + $plat)) } else { $_ } };" ^
  "Set-Content -Path $dst -Value $content;" ^
  "Write-Output $dst" ^
`) DO SET "TMP_COMPOSE=%%i"

IF NOT DEFINED TMP_COMPOSE (
  echo Error: failed to generate temporary docker-compose file.
  ENDLOCAL & EXIT /b 1
)

REM ── Overwrite docker-compose.yaml with the temp (no backup, no restore)
move /y "%TMP_COMPOSE%" "%SRC_COMPOSE%" >nul || (
  echo Error: failed to activate temporary compose file.
  IF EXIST "%TMP_COMPOSE%" del /f /q "%TMP_COMPOSE%" >nul 2>&1
  ENDLOCAL & EXIT /b 1
)
REM at this point %TMP_COMPOSE% no longer exists (it was moved/renamed)

REM ── Launch CTK container detached; capture only the last line (container id)
SET "ctk_container="
FOR /F "usebackq delims=" %%i IN (`%DC% run -d ctk 2^>^&1`) DO SET "ctk_container=%%i"

IF NOT DEFINED ctk_container (
  echo Failed to start CTK container.
  ENDLOCAL & EXIT /b 1
)

REM ── Stream logs until completion and propagate the container's exit code
CALL :stream_logs
IF ERRORLEVEL 1 GOTO :cleanup

REM ── Process generated report
SET "reportPath=reports\index.html"
SET "modifiedReportPath=REPORT.HTML"

IF EXIST "%reportPath%" (
  CALL :tweak_report "%reportPath%"
  IF ERRORLEVEL 1 echo Warning: tweak_report failed; continuing with original file.
  copy /y "%reportPath%" "%modifiedReportPath%" >nul
  START "" "%modifiedReportPath%"
) ELSE (
  echo Report file "%reportPath%" not found.
)

:cleanup
IF DEFINED ctk_container docker rm -f "%ctk_container%" >nul 2>&1
ENDLOCAL
EXIT /b


:stream_logs
REM Follow logs; then return the container's real exit code
docker logs -f "%ctk_container%"
FOR /F %%e IN ('docker wait "%ctk_container%"') DO SET "ctk_exit=%%e"
EXIT /b !ctk_exit!


:tweak_report
REM Expand collapsed sections in the HTML report
SET "reportFile=%~1"
SETLOCAL DisableDelayedExpansion
powershell -NoProfile -Command ^
  "$p = '%reportFile%';" ^
  "$c = Get-Content -Path $p -Raw;" ^
  "$c = $c -replace '.state=\{expanded:!1\}', '.state={expanded:1}';" ^
  "Set-Content -Path $p -Value $c"
SET "psrc=%ERRORLEVEL%"
ENDLOCAL & EXIT /b %psrc%


