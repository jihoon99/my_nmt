import sys

STR = '▁'

if __name__ == "__main__":
    ref_fn = sys.argv[1] # get a line from terminal : PATH

    f = open(ref_fn, 'r') # open file

    for ref in f:
        ref = ref.strip()
        input_line = sys.stdin.readline().strip() # 여기가 이해가 안되네...;ㅣㅁㄴ아ㅓㄹ;ㅣㅁㄴ얼;ㅣ만얼;ㅣ멍ㄹ;ㅣ멍;ㅣㅏ
        
        if input_line != "":
            buf = [STR]

            ref_index = 0
            input_index = 0
            while input_index < len(input_line):
                c = input_line[input_index]
                input_index += 1

                if c != ' ':
                    while ref_index < len(ref):
                        c_ = ref[ref_index]
                        ref_index += 1

                        if c_ == ' ':
                            c = STR + c
                        else:
                            break
                buf += [c]

#            # We assume that stdin has more tokens than reference input.
#            for ref_token in ref_tokens:
#                tmp_buf = []
#
#                while idx < len(tokens):
#                    if tokens[idx].strip() == '':
#                        idx += 1
#                        continue
#
#                    tmp_buf += [tokens[idx]]
#                    idx += 1
#
#                    if ''.join(tmp_buf) == ref_token:
#                        break
#
#                if len(tmp_buf) > 0:
#                    buf += [STR + tmp_buf[0].strip()] + tmp_buf[1:]

            sys.stdout.write(''.join(buf) + '\n')
        else:
            sys.stdout.write('\n')

    f.close()
