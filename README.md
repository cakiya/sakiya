# sakiya
sakiya discord bot

<img width="564" height="433" alt="image" src="https://github.com/user-attachments/assets/5c21f32a-6b7e-423e-821b-d1adc41fcd38" />

hacked together with ai assistance in 1 day for basic functionality / 5 hrs for memory features, rag, etc. on 9/21/2026

continuing to work on it whenever i feel like it ^-^

ran on i9 4060 (8gb vram) laptop

3449 lines/messages scraped persona of my chatting in a server

## setup

1. put `.gguf` models in `input/models`
2. put persona text files (eg. `cakiya.txt`) in `input/personas` (short sample included)
3. put prompt in input/personas and rename as `prompt.txt`
4. setup `.env` (example `.env_example` included)
5. download ``koboldcpp.exe`` and put it in project root folder

to use: run bot.py

## features:

### chat

1. @sakiya message/question/whatever
2. /sakiya message/question/whatever
3. dm sakiya 

### memory

sakiya's memory works with 2 different memories
1. `channel_memory`
    - if 2 people are calling it in the same channel it only knows the calls and not the non-call chats that happened between them
    - basically it only memorizes whenever calls are made to it
    - use /sync_memory if you want to reference something in chat!
2. `global_user_memory`
    - per person
    - memorizes the last 5 messages from the person that called, no matter the channel

### /sync_memory 

- absorbs the last 25 messages of the chat into `channel_memory` (max 25 messages stored in memory)

### /clear_memory

- clears both bot's `channel_memory` & `global_user_memory` of you

## todo:
1. ~~add a start script that auto loads koboldcpp.exe~~
2. ~~model select~~ set in env
3. model switcher and command?
4. personalization & mimic modes (and command to switch between them?)
5. invite link (maybe gated with a password or my own user?) command
6. ~~give the bot more context like inquiring user's username/nickname~~
7. ~~grab context of preivous x messages, and save that to memory to chat with some context?~~
8. story mode: use a differnet prompt for only story writing
9. per channel memory instead of per person memory that persists through different channels
