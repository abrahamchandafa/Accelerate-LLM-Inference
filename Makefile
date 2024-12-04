# A Minimalistic Makefile
.PHONY: all

# rename with your uid, make sure no space after your uid
UID = 3036030958

# Default target
all: prepare parallel

# download file if not found in the folder
prepare:
	@if [ ! -f model.bin ]; then \
		wget -O model.bin https://huggingface.co/huangs0/smollm/resolve/main/model.bin; \
	fi
	@if [ ! -f tokenizer.bin ]; then \
		wget -O tokenizer.bin https://huggingface.co/huangs0/smollm/resolve/main/tokenizer.bin; \
	fi

# Compile target to compile the C program
seq:
	gcc -o seq seq.c -O2 -lm

parallel:
	gcc -o parallel parallel_$(UID).c -O2 -lm -lpthread

# Clean target to remove the downloaded file
clean:
	rm -f seq parallel

clean_bin:
	rm -f model.bin tokenizer.bin