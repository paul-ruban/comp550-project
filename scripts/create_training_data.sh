#!/bin/bash

###################################################################################
# THIS IS A TEMPLATE SCRIPT FOR RUNNING SIMPLE PROGRAMS AND LAUNCHING EXPERIMENTS #
# MAIN PARTS: * PARSING ARGUMENTS, LIKE (INPUT, OUTPUT DIRS, FLAGS OR H-PARAMS)   #
#             * VALIDATING ARGUMENTS AND PROCESSING THEM IF REQUIRED              #
#             * EXECUTING THE PROGRAM OR LAUNCHING EXPERIMENT                     #
#             * EXAMPLE OF INVOCATION, SO IT CAN BE EDDITED AND/OR EXECUTED       #
#                                                                                 # 
# PS: WHEN CREATING SCRIPTS DON'T FORGET TO MAKE THEM EXECUTABLE                  #
#     EX. chmod +x template.sh (allows executable permissions to template.sh)     #
###################################################################################


############### INFER THE SCRIPT DIRNAME ###############

prg=$0

if [ ! -e "${prg}" ]; then
    case $prg in
        (*/*) exit 1;;
        (*) prg=$(command -v -- "${prg}") || exit;;
    esac
fi

dir=$(
    cd -P -- "$(dirname -- "${prg}")" && pwd -P
) || exit
prg=${dir}/$(basename -- "${prg}") || exit


############### PARSE ARGS ###############

for i in "$@"
do
case $i in
    ###############
    --input_dir=*)
    input_dir="${i#*=}"
    shift
    ;;
    ###############
    --output_dir=*)
    output_dir="${i#*=}"
    shift
    ;;
    ###############
    # add more arguments
    *)
    # ignore unknown options
    ;;
esac
done

############### VALIDATE AND PROCESS ARGS ###############

if [ -z "${input_dir}" ]; then
    echo "ERROR: missing argument --input_dir"
    exit 1
fi

if [ -z "${output_dir}" ]; then
    echo "ERROR: missing argument --output_dir"
    exit 1
fi


############### EXECUTE PROGRAM ###############

tmp_file=${output_dir}/temp.tmp
# remove temporary file if exists
rm -f ${tmp_file}

for f in ${input_dir}/*.txt
do
    s='START OF THE PROJECT GUTENBERG EBOOK'
    e='END OF THE PROJECT GUTENBERG EBOOK'

    cat $f |
    tr '\t' ' ' |
    tr -d '\r' |
    tr '\r\n' '\v' |
    sed -e 's/[ ]*\v[ ]*/\v/g' |
    sed -E 's/[\v]{2,}/\n/g' |
    sed 's/\v/ /g' |
    sed -n "/$s/, /$e/{ /$s/! { /$e/! p } }" |
    grep '\S' |
    tr '[:upper:]' '[:lower:]' |
    sed 's/<[^>]*>//g' \
    >> ${tmp_file}
done

exit $? # end of program, return the most resent value


############### COMMAND TO RUN THE PROGRAM ###############

./create_training_data.sh\
    --input_dir=/home/pavlo/comp-550/comp-550-project/data \
    --output_dir=/home/pavlo/comp-550/comp-550-project/data