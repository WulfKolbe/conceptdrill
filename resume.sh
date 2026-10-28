#!/usr/bin/env bash
# Resume the conceptdrill working session.
printf '\033]1;%s\007' 'iCOCNEPTDRILL'
# Set the window title
printf '\033]2;%s\007' 'CONCEPTDRILL'
export CLAUDE_CODE_DISABLE_TERMINAL_TITLE=1
SESSION_ID=4b8925f0-dbcf-4ad2-b216-50011f79e45f
claude --resume "$SESSION_ID" --dangerously-skip-permissions
