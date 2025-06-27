#!/bin/bash

usage()
{
  echo "Usage: $0 <algorithm> <iva> <iva data> <iva data file> <core count file> <power profile file> <time serial analytics file> <time parallel analytics file> <space serial analytics file> <space parallel analytics file> <power serial analytics file> <power parallel analytics file> <energy serial analytics file> <energy parallel analytics file> <speedup analytics file> <freeup analytics file> <powerup analytics file> <energyup analytics file> <id> <repo> <repo name> <start time> <progress>"
  exit 1
}

check_abort() {
  local repo_path=$1
  local abort_file="$repo_path/do.abort"

  echo "[$$] Checking for abort signal in $abort_file"

  if [[ ! -f "$abort_file" ]]; then
    return 1
  fi

  local flag=$(cat "$abort_file" 2>/dev/null | tr -d '[:space]' | tr '[:upper]' '[:lower]')
  echo "[$$] Abort signal found: $flag"

  if [[ "$flag" == "1" || "$flag" == "true" || "$flag" == "abort" ]]; then
    echo "[$$] abort signal detected, terminating.."

    # if [[ -d "$repo_path" ]]; then
    #   rm -rf "$repo_path"
    #   echo "[$$] Repository directory cleaned up : $repo_path"
    # fi

    return 0
  fi

  return 1
}

call_fit() {
  local in_file=$1
  local out_file=$2
  local progress=$3
  local progress_bandwidth=$4
  local fit_count=$5
  local id=$6
  local repo=$7
  local repo_name=$8
  local start_time=$9
  local analysis_file=${10}

  fit.py --in-file "${1}" --out-file "${2}"

  progress=`echo "scale=1; p=$progress; bw=$progress_bandwidth; l=$fit_count; p + (bw/l)" | bc -l`

  echo "{\"id\":\"$id\",\"repo\":\"$repo\",\"repoName\":\"$repo_name\",\"startTime\":\"$start_time\",\
  \"endTime\":\"\",\"status\":\"In progress\",\"progress\":{\"currentStep\":\"Predictive Model Generation\",\
  \"nextStep\":\"None\",\"percent\":$progress},\
  \"result\":{\"errorCode\":0,\"message\":\"\",\"repo\":\"\"}}" > $analysis_file
}

if [ "$#" -ne 29 ]; then
    echo "Invalid number of parameters. Expected:30 Passed:$#"
    usage
fi

algo=$1
main_file=$2
target_fn=$3
target_fn_iva_name=$4
target_fn_iva_start=$5
target_fn_iva_end=$6
argc=$7
iva_name=$8
iva_data=$9
iva_data_file=${10}
core_count_file=${11}
power_profile_file=${12}
time_serial_analytics_file=${13}
time_parallel_analytics_file=${14}
# time_parallel_slow_analytics_file=${15}
space_serial_analytics_file=${15}
space_parallel_analytics_file=${16}
power_serial_analytics_file=${17}
power_parallel_analytics_file=${18}
energy_serial_analytics_file=${19}
energy_parallel_analytics_file=${20}
speedup_analytics_file=${21}
freeup_analytics_file=${22}
powerup_analytics_file=${23}
energyup_analytics_file=${24}
id=${25}
repo=${26}
repo_name=${27}
start_time=${28}
progress=${29}

serial_measurement=serial.csv
parallel_measurement=parallel.csv
# parallel_slow_measurement=parallel_slow.csv
analysis_file=analysis.json

# parallel code generation config
parallel_plugin_so=MyRewriter.so
parallel_plugin_name=rew

echo "cleaning up"

# cleanup
rm -f $time_serial_analytics_file $time_parallel_analytics_file $time_parallel_slow_analytics_file $space_serial_analytics_file \
   $space_parallel_analytics_file $power_serial_analytics_file $power_parallel_analytics_file \
   $energy_serial_analytics_file $energy_parallel_analytics_file $speedup_analytics_file \
   $freeup_analytics_file $powerup_analytics_file $energyup_analytics_file \
   $serial_measurement $parallel_measurement $parallel_slow_measurement

echo "cleanup done"

readarray -t iva_arr  < $iva_data_file
readarray -t core_arr < $core_count_file

power_profile=()

while IFS=, read -r i p;
do power_profile+=($p);
done < $power_profile_file

iva=()
core=()

for i in ${iva_arr[@]}
do
  iva+=($i)
done

for i in ${core_arr[@]}
do
  core+=($i)
done

# make - serial
make -f Makefile-serial

# make a copy of original main file
main_file_extn="${main_file##*.}"
main_file_noextn="${main_file%.*}"
main_file_orig="$main_file_noextn"_original."$main_file_extn"
#cp $main_file $main_file_orig

## make a copy of original execuatble
algo_orig="$algo"_original
#mv $algo $algo_orig

# generate TALP parallel code
#clang -fplugin=$parallel_plugin_so -Xclang -plugin -Xclang $parallel_plugin_name -Xclang -plugin-arg-rew -Xclang -target-function -Xclang -plugin-arg-rew -Xclang $target_fn -Xclang -plugin-arg-rew -Xclang -out-file -Xclang -plugin-arg-rew -Xclang $main_file -Xclang -plugin-arg-rew -Xclang -iva -Xclang -plugin-arg-rew -Xclang $target_fn_iva_name -Xclang -plugin-arg-rew -Xclang -iva-start -Xclang -plugin-arg-rew -Xclang $target_fn_iva_start -Xclang -plugin-arg-rew -Xclang -iva-end -Xclang -plugin-arg-rew -Xclang $target_fn_iva_end -Xclang -plugin-arg-rew -Xclang -argc -Xclang -plugin-arg-rew -Xclang $argc -c $main_file

# make - parallel
make -f Makefile-parallel

# serial run

time_serial=()
space_serial=()
power_serial=()
energy_serial=()

repo_path="/app/data/repo-import/$repo_name"

if check_abort $repo_path; then exit 2; fi

# time - serial
progress_bandwidth=10

for i in ${iva[@]}
do
  # time
  start=`date +%s.%N`;\
  ./$algo_orig $i;\
  end=`date +%s.%N`;\
  time_serial+=(`printf '%.8f' $( echo "$end - $start" | bc -l )`);

  progress=`echo "scale=1; p=$progress; bw=$progress_bandwidth; l=${#iva[@]}; p + (bw/l)" | bc -l`

  echo "{\"id\":\"$id\",\"repo\":\"$repo\",\"repoName\":\"$repo_name\",\"startTime\":\"$start_time\",\
  \"endTime\":\"\",\"status\":\"In progress\",\"progress\":{\"currentStep\":\"Serial Time Measurement\",\
  \"nextStep\":\"Serial Memory Measurement\",\"percent\":$progress},\
  \"result\":{\"errorCode\":0,\"message\":\"\",\"repo\":\"\"}}" > $analysis_file
done

if check_abort $repo_path; then exit 2; fi

# memory - serial
progress_bandwidth=10

count=1
for i in ${iva[@]}
do
  # memory
  heaptrack -o "$algo.$count" ./$algo_orig $i;\
  space_serial+=(`heaptrack --analyze "$algo.$count.zst"  | grep "peak heap memory consumption" | awk '{print $5}'`);
  count=$((count+1))

  progress=`echo "scale=1; p=$progress; bw=$progress_bandwidth; l=${#iva[@]}; p + (bw/l)" | bc -l`

  echo "{\"id\":\"$id\",\"repo\":\"$repo\",\"repoName\":\"$repo_name\",\"startTime\":\"$start_time\",\
  \"endTime\":\"\",\"status\":\"In progress\",\"progress\":{\"currentStep\":\"Serial Memory Measurement\",\
  \"nextStep\":\"Serial Power Measurement\",\"percent\":$progress},\
  \"result\":{\"errorCode\":0,\"message\":\"\",\"repo\":\"\"}}" > $analysis_file

done

if check_abort $repo_path; then exit 2; fi

# power - serial
progress_bandwidth=10

for i in ${iva[@]}
do
  # power
  power_serial+=(${power_profile[0]})

  progress=`echo "scale=1; p=$progress; bw=$progress_bandwidth; l=${#iva[@]}; p + (bw/l)" | bc -l`

  echo "{\"id\":\"$id\",\"repo\":\"$repo\",\"repoName\":\"$repo_name\",\"startTime\":\"$start_time\",\
  \"endTime\":\"\",\"status\":\"In progress\",\"progress\":{\"currentStep\":\"Serial Power Measurement\",\
  \"nextStep\":\"Parallel Time Measurement\",\"percent\":$progress},\
  \"result\":{\"errorCode\":0,\"message\":\"\",\"repo\":\"\"}}" > $analysis_file

done

# energy - serial
for i in "${!iva[@]}"
do
  # energy
  energy_serial+=(`echo "tm=${time_serial[i]};pw=${power_serial[i]};tm * pw" | bc -l`);
done

# serial measurement file
for i in "${!iva[@]}"
do
  echo "${iva[i]},${time_serial[i]},${space_serial[i]},${power_serial[i]},${energy_serial[i]}" >> "$serial_measurement"
done

# parallel run

time_parallel=()
space_parallel=()
power_parallel=()
energy_parallel=()

# Add new parallel slow measurement arrays
# time_parallel_slow=()

# time - parallel
progress_bandwidth=10

for i in ${core[@]}
do
  # time
  start=`date +%s.%N`;\
  curl -D - --header "Content-Type: application/json" --output - --request POST --data '{"id": 1, "lib": "libfft.so", "core": '"$i"', "argv": ["main", '\""$iva_data"\"','\""$i"\"']}' 192.168.1.36:7092/run;\
  end=`date +%s.%N`;\
  time_parallel+=(`printf '%.8f' $( echo "$end - $start" | bc -l )`);

  progress=`echo "scale=1; p=$progress; bw=$progress_bandwidth; l=${#core[@]}; p + (bw/l)" | bc -l`

  echo "{\"id\":\"$id\",\"repo\":\"$repo\",\"repoName\":\"$repo_name\",\"startTime\":\"$start_time\",\
  \"endTime\":\"\",\"status\":\"In progress\",\"progress\":{\"currentStep\":\"Parallel Time Measurement\",\
  \"nextStep\":\"Parallel Memory Measurement\",\"percent\":$progress},\
  \"result\":{\"errorCode\":0,\"message\":\"\",\"repo\":\"\"}}" > $analysis_file
done

if check_abort $repo_path; then exit 2; fi

# memory - parallel
progress_bandwidth=10

count=1
for i in ${core[@]}
do
  # memory
  heaptrack -o "$algo.$count" ./$algo $iva_data $i;\
  space_parallel+=(`heaptrack --analyze "$algo.$count.zst"  | grep "peak heap memory consumption" | awk '{print $5}'`);
  count=$((count+1))

  progress=`echo "scale=1; p=$progress; bw=$progress_bandwidth; l=${#core[@]}; p + (bw/l)" | bc -l`

  echo "{\"id\":\"$id\",\"repo\":\"$repo\",\"repoName\":\"$repo_name\",\"startTime\":\"$start_time\",\
  \"endTime\":\"\",\"status\":\"In progress\",\"progress\":{\"currentStep\":\"Parallel Memory Measurement\",\
  \"nextStep\":\"Parallel Power Measurement\",\"percent\":$progress},\
  \"result\":{\"errorCode\":0,\"message\":\"\",\"repo\":\"\"}}" > $analysis_file

done

# power - parallel
progress_bandwidth=10

for i in ${core[@]}
do
  # power
  power_parallel+=(${power_profile[i-1]})

  progress=`echo "scale=1; p=$progress; bw=$progress_bandwidth; l=${#core[@]}; p + (bw/l)" | bc -l`

  echo "{\"id\":\"$id\",\"repo\":\"$repo\",\"repoName\":\"$repo_name\",\"startTime\":\"$start_time\",\
  \"endTime\":\"\",\"status\":\"In progress\",\"progress\":{\"currentStep\":\"Parallel Power Measurement\",\
  \"nextStep\":\"Predictive Model Generation\",\"percent\":$progress},\
  \"result\":{\"errorCode\":0,\"message\":\"\",\"repo\":\"\"}}" > $analysis_file
done

# energy - parallel
for i in "${!core[@]}"
do
  # energy
  energy_parallel+=(`echo "tm=${time_parallel[i]};pw=${power_parallel[i]};tm * pw" | bc -l`);
done

if check_abort $repo_path; then exit 2; fi

# parallel measurement file
for i in "${!core[@]}"
do
  echo "${core[i]},${time_parallel[i]},${space_parallel[i]},${power_parallel[i]},${energy_parallel[i]}" >> "$parallel_measurement"
done

if check_abort $repo_path; then exit 2; fi

# parallel slow measurement file
# for i in "${!core[@]}"
# do
#   echo "${core[i]},${time_parallel_slow[i]}" >> "$parallel_slow_measurement"
# done

if check_abort $repo_path; then exit 2; fi

# data prep
for i in "${!space_serial[@]}"; do
  if [[ ${space_serial[$i]: -1} == "K" ]]; then
    val=${space_serial[$i]::-1}
    space_serial[$i]=`printf '%.4f' $(echo "v=$val;v * 0.001" | bc -l)`
  else
    space_serial[$i]=${space_serial[$i]::-1}
  fi
done

for i in "${!space_parallel[@]}"; do
  if [[ ${space_parallel[$i]: -1} == "K" ]]; then
    val=${space_parallel[$i]::-1}
    space_parallel[$i]=`printf '%.4f' $(echo "v=$val;v * 0.001" | bc -l)`
  else
    space_parallel[$i]=${space_parallel[$i]::-1}
  fi
done

# speedup
speedup=()
t_max=${time_parallel[0]}
for t in "${time_parallel[@]}"; do
  speedup+=(`echo "scale=2;$t_max/$t" | bc -l`)
done

# freeup
freeup=()
s_max=${space_parallel[0]}
for s in "${space_parallel[@]}"; do
  freeup+=(`echo "scale=2;$s_max/$s" | bc -l`)
done

# powerup
powerup=()
p_1=${power_parallel[0]}
for p_core in "${power_parallel[@]}"; do
  powerup+=(`echo "scale=4;$p_1/$p_core" | bc -l`)
done

# energyup
energyup=()
e_1=${energy_parallel[0]}
for e_core in "${energy_parallel[@]}"; do
  energyup+=(`echo "scale=4;$e_1/$e_core" | bc -l`)
done

jo -p iva=$(jo name=$iva_name values=$(jo -a ${iva[@]})) \
measurements=$(jo -a ${time_serial[@]}) > time-serial.json
jo -p iva=$(jo name=core values=$(jo -a ${core[@]})) \
measurements=$(jo -a ${time_parallel[@]}) > time-parallel.json
jo -p iva=$(jo name=$iva_name values=$(jo -a ${iva[@]})) \
measurements=$(jo -a ${space_serial[@]}) > space-serial.json
jo -p iva=$(jo name=core values=$(jo -a ${core[@]})) \
measurements=$(jo -a ${space_parallel[@]}) > space-parallel.json
jo -p iva=$(jo name=$iva_name values=$(jo -a ${iva[@]})) \
measurements=$(jo -a ${power_serial[@]}) > power-serial.json
jo -p iva=$(jo name=core values=$(jo -a ${core[@]})) \
measurements=$(jo -a ${power_parallel[@]}) > power-parallel.json
jo -p iva=$(jo name=$iva_name values=$(jo -a ${iva[@]})) \
measurements=$(jo -a ${energy_serial[@]}) > energy-serial.json
jo -p iva=$(jo name=core values=$(jo -a ${core[@]})) \
measurements=$(jo -a ${energy_parallel[@]}) > energy-parallel.json
jo -p iva=$(jo name=core values=$(jo -a ${core[@]})) \
measurements=$(jo -a ${speedup[@]}) > speedup.json
jo -p iva=$(jo name=core values=$(jo -a ${core[@]})) \
measurements=$(jo -a ${freeup[@]}) > freeup.json
jo -p iva=$(jo name=core values=$(jo -a ${core[@]})) \
measurements=$(jo -a ${powerup[@]}) > powerup.json
jo -p iva=$(jo name=core values=$(jo -a ${core[@]})) \
measurements=$(jo -a ${energyup[@]}) > energyup.json

# curve fitting

progress_bandwidth=10
fit_count=13
analysis_types=('time-serial' 'time-parallel' 'space-serial' 'space-parallel' 'power-serial' 'power-parallel' 'energy-serial' 'energy-parallel' 'speedup' 'freeup' 'powerup' 'energyup')

for i in "${analysis_types[@]}"
do
  echo "${i}.json"
  echo "${i}-fitted.json"
  call_fit $i.json $i-fitted.json $progress $progress_bandwidth $fit_count $id $repo $repo_name $start_time $analysis_file
done

# time serial
extn="${time_serial_analytics_file##*.}"
noextn="${time_serial_analytics_file%.*}"

time_serial_analytics_file_d="$noextn"."$extn"

jo -p \
iva=$(jo data=$(jo -a ${iva[@]}) name=$iva_name unit=size) \
measurements=$(jo data=$(jo -a ${time_serial[@]}) name=time unit=seconds) \
fitted=$(jo data="`jq '.fitted' time-serial-fitted.json`" name=time unit=seconds) \
unoptimized=$(jo data=$(jo -a) name=time unit=seconds) \
fit_method="`jq -r '.method' time-serial-fitted.json`" \
mse="`jq '.mse' time-serial-fitted.json`" \
> $time_serial_analytics_file_d

# time parallel
extn="${time_parallel_analytics_file##*.}"
noextn="${time_parallel_analytics_file%.*}"

time_parallel_analytics_file_d="$noextn"."$extn"

jo -p \
iva=$(jo data=$(jo -a ${core[@]}) name=core unit=count) \
measurements=$(jo data=$(jo -a ${time_parallel[@]}) name=time unit=seconds) \
fitted=$(jo data="`jq '.fitted' time-parallel-fitted.json`" name=time unit=seconds) \
fit_method="`jq -r '.method' time-parallel-fitted.json`" \
mse="`jq '.mse' time-parallel-fitted.json`" \
> $time_parallel_analytics_file_d

# memory serial
extn="${space_serial_analytics_file##*.}"
noextn="${space_serial_analytics_file%.*}"

space_serial_analytics_file_d="$noextn"."$extn"

jo -p \
iva=$(jo data=$(jo -a ${iva[@]}) name=$iva_name unit=size) \
measurements=$(jo data=$(jo -a ${space_serial[@]}) name=memory unit=MB) \
unoptimized=$(jo data=$(jo -a) name=memory unit=MB) \
fitted=$(jo data="`jq '.fitted' space-serial-fitted.json`" name=memory unit=MB) \
fit_method="`jq -r '.method' space-serial-fitted.json`" \
mse="`jq '.mse' space-serial-fitted.json`" \
> $space_serial_analytics_file_d

# memory parallel
extn="${space_parallel_analytics_file##*.}"
noextn="${space_parallel_analytics_file%.*}"

space_parallel_analytics_file_d="$noextn"."$extn"

jo -p \
iva=$(jo data=$(jo -a ${core[@]}) name=core unit=count) \
measurements=$(jo data=$(jo -a ${space_parallel[@]}) name=memory unit=MB) \
unoptimized=$(jo data=$(jo -a) name=memory unit=MB) \
fitted=$(jo data="`jq '.fitted' space-parallel-fitted.json`" name=memory unit=MB) \
fit_method="`jq -r '.method' space-parallel-fitted.json`" \
mse="`jq '.mse' space-parallel-fitted.json`" \
> $space_parallel_analytics_file_d

# power serial
extn="${power_serial_analytics_file##*.}"
noextn="${power_serial_analytics_file%.*}"

power_serial_analytics_file_d="$noextn"."$extn"

jo -p \
iva=$(jo data=$(jo -a ${iva[@]}) name=$iva_name unit=size) \
measurements=$(jo data=$(jo -a ${power_serial[@]}) name=power unit="watts") \
unoptimized=$(jo data=$(jo -a) name=power unit="watts") \
fitted=$(jo data="`jq '.fitted' power-serial-fitted.json`" name=power unit="watts") \
fit_method="`jq -r '.method' power-serial-fitted.json`" \
mse="`jq '.mse' power-serial-fitted.json`" \
> $power_serial_analytics_file_d

# power parallel
extn="${power_parallel_analytics_file##*.}"
noextn="${power_parallel_analytics_file%.*}"

power_parallel_analytics_file_d="$noextn"."$extn"

jo -p \
iva=$(jo data=$(jo -a ${core[@]}) name=core unit=count) \
measurements=$(jo data=$(jo -a ${power_parallel[@]}) name=power unit="watts") \
fitted=$(jo data="`jq '.fitted' power-parallel-fitted.json`" name=power unit="watts") \
unoptimized=$(jo data=$(jo -a) name=power unit="watts") \
fit_method="`jq -r '.method' power-parallel-fitted.json`" \
mse="`jq '.mse' power-parallel-fitted.json`" \
> $power_parallel_analytics_file_d

# energy serial
extn="${energy_serial_analytics_file##*.}"
noextn="${energy_serial_analytics_file%.*}"

energy_serial_analytics_file_d="$noextn"."$extn"

jo -p \
iva=$(jo data=$(jo -a ${iva[@]}) name=$iva_name unit=size) \
measurements=$(jo data=$(jo -a ${energy_serial[@]}) name=energy unit="watt-seconds") \
unoptimized=$(jo data=$(jo -a) name=energy unit="watt-seconds") \
fitted=$(jo data="`jq '.fitted' energy-serial-fitted.json`" name=energy unit="watt-seconds") \
fit_method="`jq -r '.method' energy-serial-fitted.json`" \
mse="`jq '.mse' energy-serial-fitted.json`" \
> $energy_serial_analytics_file_d

# energy parallel
extn="${energy_parallel_analytics_file##*.}"
noextn="${energy_parallel_analytics_file%.*}"

energy_parallel_analytics_file_d="$noextn"."$extn"

jo -p \
iva=$(jo data=$(jo -a ${core[@]}) name=core unit=count) \
measurements=$(jo data=$(jo -a ${energy_parallel[@]}) name=energy unit="watt-seconds") \
unoptimized=$(jo data=$(jo -a) name=energy unit="watt-seconds") \
fitted=$(jo data="`jq '.fitted' energy-parallel-fitted.json`" name=energy unit="watt-seconds") \
fit_method="`jq -r '.method' energy-parallel-fitted.json`" \
mse="`jq '.mse' energy-parallel-fitted.json`" \
> $energy_parallel_analytics_file_d

# speedup
extn="${speedup_analytics_file##*.}"
noextn="${speedup_analytics_file%.*}"

speedup_analytics_file_d="$noextn"."$extn"

jo -p \
iva=$(jo data=$(jo -a ${core[@]}) name=core unit=count) \
measurements=$(jo data=$(jo -a ${speedup[@]}) name='T1/Tcore' unit='') \
unoptimized=$(jo data=$(jo -a) name='T1/Tcore' unit='') \
fitted=$(jo data="`jq '.fitted' speedup-fitted.json`" name='T1/Tcore' unit='') \
fit_method="`jq -r '.method' speedup-fitted.json`" \
mse="`jq '.mse' speedup-fitted.json`" \
> $speedup_analytics_file_d

# freeup
extn="${freeup_analytics_file##*.}"
noextn="${freeup_analytics_file%.*}"

freeup_analytics_file_d="$noextn"."$extn"

jo -p \
iva=$(jo data=$(jo -a ${core[@]}) name=core unit=count) \
measurements=$(jo data=$(jo -a ${freeup[@]}) name='S1/Score' unit='') \
unoptimized=$(jo data=$(jo -a) name='S1/Score' unit='') \
fitted=$(jo data="`jq '.fitted' freeup-fitted.json`" name='S1/Score' unit='') \
fit_method="`jq -r '.method' freeup-fitted.json`" \
mse="`jq '.mse' freeup-fitted.json`" \
> $freeup_analytics_file_d

# powerup
extn="${powerup_analytics_file##*.}"
noextn="${powerup_analytics_file%.*}"

powerup_analytics_file_d="$noextn"."$extn"

jo -p \
iva=$(jo data=$(jo -a ${core[@]}) name=core unit=count) \
measurements=$(jo data=$(jo -a ${powerup[@]}) name='PowerEfficiency(P1/Pcore)' unit='') \
unoptimized=$(jo data=$(jo -a) name='PowerEfficiency(P1/Pcore)' unit='') \
fitted=$(jo data="`jq '.fitted' powerup-fitted.json`" name='PowerEfficiency(P1/Pcore)' unit='') \
fit_method="`jq -r '.method' powerup-fitted.json`" \
mse="`jq '.mse' powerup-fitted.json`" \
> $powerup_analytics_file_d

# energyup
extn="${energyup_analytics_file##*.}"
noextn="${energyup_analytics_file%.*}"

energyup_analytics_file_d="$noextn"."$extn"

jo -p \
iva=$(jo data=$(jo -a ${core[@]}) name=core unit=count) \
measurements=$(jo data=$(jo -a ${energyup[@]}) name='EnergyEfficiency(E1/Ecore)' unit='') \
unoptimized=$(jo data=$(jo -a) name='EnergyEfficiency(E1/Ecore)' unit='') \
fitted=$(jo data="`jq '.fitted' energyup-fitted.json`" name='EnergyEfficiency(E1/Ecore)' unit='') \
fit_method="`jq -r '.method' energyup-fitted.json`" \
mse="`jq '.mse' energyup-fitted.json`" \
> $energyup_analytics_file_d