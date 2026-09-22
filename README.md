# sakiya
sakiya discord bot

<img width="564" height="433" alt="image" src="https://github.com/user-attachments/assets/5c21f32a-6b7e-423e-821b-d1adc41fcd38" />

ran on i9 4060 (8gb vram) laptop

3449 lines/messages scraped persona of my chatting in a server

RAG turned off to keep personality consistent, my laptop not powerful enough to include enough context that it gets back to consistent...



## setup

1. put `.gguf` models in `input/models`
2. put persona text files (eg. `cakiya.txt`) in `input/personas` (short sample included)
3. put prompt in input/personas and rename as `prompt.txt`
4. setup `.env` (example `.env_example` included)

to use: run bot.py

## features:

chat

1. @sakiya message/question/whatever
2. /sakiya message/question/whatever
3. dm sakiya 

/sync_memory 

- absorbs the last 25 messages of the chat into memory (max 25 messages memory)

/clear_memory

- clears only the bot's memory of you

note: rag works but currently disabled as it selects randomly to not overwhelm the small local llm model, which makes the personality change randomly each run

## todo:
1. ~~add a start script that auto loads koboldcpp.exe~~
2. ~~model select~~ set in env
3. model switcher and command?
4. personalization & mimic modes (and command to switch between them?)
5. invite link (maybe gated with a password or my own user?) command
6. give the bot more context like inquiring user's username/nickname
7. ~~grab context of preivous x messages, and save that to memory to chat with some context?~~
8. story mode: use a differnet prompt for only story writing
9. per channel memory instead of per person memory that persists through different channels
