#!/bin/bash
# ============================================================================
# COMPREHENSIVE SCALING ANALYSIS - MPI vs MPI+OMP
# ============================================================================
# Questo script esegue test completi per confrontare:
# 1. MPI PURO (solo processi MPI)
# 2. MPI+OMP HYBRID (bilanciato MPI x OMP)
#
# Per ogni configurazione valuta:
# - Strong Scaling (problema fisso, aumento cores)
# - Weak Scaling (problema scala con cores)
# - Efficiency (percentuale di utilizzo)
# ============================================================================

# Parametri configurabili
SIZE_MODE=$1
if [ -z "$SIZE_MODE" ]; then
    echo "ERRORE: Specifica small, medium o large"
    echo "Usage: $0 [small|medium|large]"
    exit 1
fi

# ================= CONFIGURAZIONE GLOBALE =================
CSV_FILE="scaling_comprehensive_${SIZE_MODE}.csv"
RESULTS_DIR="results_comprehensive_${SIZE_MODE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${RESULTS_DIR}

# Inizializza CSV con header
echo "Size,ScalingType,Mode,Cores,MPI_Ranks,OMP_Threads,Nodes,Time_ms,Speedup,Efficiency,ProblemSize" > $CSV_FILE

BASE_FILE="test_files/test_02_a30k_p20k_w6"
LINES_PER_FILE=20000
EXE="$PWD/energy_storms_mpi_omp"

# Verifica esistenza eseguibile
if [ ! -f "$EXE" ]; then
    echo "ERRORE: Eseguibile $EXE non trovato!"
    exit 1
fi

# ================= CONFIGURAZIONI SIZE =================
# IMPORTANTE: Tutti e 3 i size testano TUTTI i core: 1,4,8,16,32,64,128
# Anche se l'efficienza decade, serve per vedere cosa va male

if [ "$SIZE_MODE" == "small" ]; then
    BASE_LAYER_SIZE=100000
    BASE_TARGET_PARTICLES=60000
    CORES_TO_TEST=(1 4 8 16 32 64)

elif [ "$SIZE_MODE" == "medium" ]; then
    BASE_LAYER_SIZE=500000
    BASE_TARGET_PARTICLES=160000
    CORES_TO_TEST=(1 4 8 16 32 64)

elif [ "$SIZE_MODE" == "large" ]; then
    BASE_LAYER_SIZE=2000000
    BASE_TARGET_PARTICLES=300000
    CORES_TO_TEST=(1 4 8 16 32 64)
else
    echo "ERRORE: SIZE_MODE deve essere small, medium o large"
    exit 1
fi

echo "========================================"
echo "COMPREHENSIVE SCALING ANALYSIS - $SIZE_MODE"
echo "========================================"
echo "Base layer size: $BASE_LAYER_SIZE"
echo "Base target particles: $BASE_TARGET_PARTICLES"
echo "Cores to test: ${CORES_TO_TEST[@]}"
echo "Testing modes: MPI-Pure, MPI+OMP-Hybrid"
echo "Scaling types: Strong, Weak"
echo "========================================"

# ================= TROVA FILE DISPONIBILI =================
AVAILABLE_FILES=()
for i in {1..100}; do
    file="${BASE_FILE}_${i}.txt"
    if [ -f "$file" ]; then
        AVAILABLE_FILES+=("$file")
    fi
done

# Prova anche pattern alternativi
for suffix in 6 5 4 3 2 1; do
    file="${BASE_FILE:0:-1}${suffix}"
    if [ -f "$file" ] && [[ ! " ${AVAILABLE_FILES[@]} " =~ " ${file} " ]]; then
        AVAILABLE_FILES+=("$file")
    fi
done

if [ ${#AVAILABLE_FILES[@]} -eq 0 ]; then
    echo "ERRORE: Nessun file di input trovato!"
    exit 1
fi

echo ""
echo "Files disponibili: ${#AVAILABLE_FILES[@]}"
for f in "${AVAILABLE_FILES[@]}"; do
    echo "  - $f"
done
echo ""

# ================= FUNZIONI HELPER =================

# Funzione per generare input files ripetuti
generate_input_files() {
    local target_particles=$1
    local particles_per_set=$((${#AVAILABLE_FILES[@]} * LINES_PER_FILE))
    local repetitions=$(( (target_particles + particles_per_set - 1) / particles_per_set ))

    local input_files=""
    local total_particles=0

    for rep in $(seq 1 $repetitions); do
        for file in "${AVAILABLE_FILES[@]}"; do
            input_files="$input_files $file"
            total_particles=$((total_particles + LINES_PER_FILE))

            if [ $total_particles -ge $target_particles ]; then
                echo "$input_files"
                return
            fi
        done
    done

    echo "$input_files"
}

# Funzione per ottenere configurazione MPI+OMP ottimale
get_hybrid_config() {
    local cores=$1
    local mpi omp nodes

    case $cores in
        1)   mpi=1;  omp=1;  nodes=1 ;;
        4)   mpi=4;  omp=1;  nodes=1 ;;   # Ancora MPI puro per pochi cores
        8)   mpi=8;  omp=1;  nodes=1 ;;
        16)  mpi=16; omp=1;  nodes=1 ;;
        32)  mpi=16; omp=2;  nodes=1 ;;   # Inizio ibrido
        64)  mpi=32; omp=2;  nodes=2 ;;   # Mantieni ~32 MPI ranks
        128) mpi=32; omp=4;  nodes=4 ;;   # Mantieni ~32 MPI ranks
        *)   mpi=$cores; omp=1; nodes=1 ;;
    esac

    echo "$mpi $omp $nodes"
}

# Funzione per calcolare layer size per weak scaling
get_weak_scaling_layer_size() {
    local cores=$1
    local base_size=$BASE_LAYER_SIZE

    # Weak scaling: layer_size scala linearmente con cores
    echo $((base_size * cores))
}

# Funzione per calcolare particles per weak scaling
get_weak_scaling_particles() {
    local cores=$1
    local base_particles=$BASE_TARGET_PARTICLES

    # Weak scaling: particles scala linearmente con cores
    echo $((base_particles * cores))
}

# Variabile globale per baseline time (strong scaling)
BASELINE_TIME_STRONG=0
BASELINE_TIME_WEAK=0

# ================= FUNZIONE PRINCIPALE DI TEST =================
run_test() {
    local cores=$1
    local mpi=$2
    local omp=$3
    local nodes=$4
    local mode=$5              # "MPI-Pure" o "Hybrid"
    local scaling_type=$6      # "Strong" o "Weak"
    local layer_size=$7
    local target_particles=$8

    # Genera nome file output
    local output_file="${RESULTS_DIR}/${SIZE_MODE}_${scaling_type}_${mode}_${cores}cores_${mpi}x${omp}.txt"

    echo "" | tee -a $output_file
    echo "=== Test: $scaling_type Scaling - $mode ===" | tee -a $output_file
    echo "Cores: $cores ($mpi MPI x $omp OMP) - Nodes: $nodes" | tee -a $output_file
    echo "Layer size: $layer_size | Particles: $target_particles" | tee -a $output_file

    # Genera input files
    local input_files=$(generate_input_files $target_particles)

    # Configurazione ambiente OpenMP
    export OMP_NUM_THREADS=$omp
    export OMP_PLACES=cores
    export OMP_PROC_BIND=close

    # Esegui test con timing
    local start=$(date +%s%N)

    if [ $mpi -eq 1 ]; then
        # Caso sequenziale o solo OpenMP
        /usr/bin/time -v $EXE $layer_size $input_files 2>&1 | tee -a $output_file
    else
        # Caso MPI
        /usr/bin/time -v mpirun -np $mpi \
            --map-by ppr:$((mpi/nodes)):node \
            --bind-to core \
            $EXE $layer_size $input_files 2>&1 | tee -a $output_file
    fi

    local exit_code=${PIPESTATUS[0]}
    local end=$(date +%s%N)

    if [ $exit_code -ne 0 ]; then
        echo "ERRORE: Test fallito!" | tee -a $output_file
        return 1
    fi

    local time_ms=$(( ($end - $start) / 1000000 ))

    echo "Time: ${time_ms} ms" | tee -a $output_file

    # Calcola metriche di scaling
    local speedup efficiency

    if [ "$scaling_type" == "Strong" ]; then
        # Strong scaling: confronta con baseline (1 core, stesso problema)
        if [ $BASELINE_TIME_STRONG -eq 0 ]; then
            BASELINE_TIME_STRONG=$time_ms
            speedup=1.00
            efficiency=100.00
        else
            speedup=$(awk "BEGIN {printf \"%.2f\", $BASELINE_TIME_STRONG / $time_ms}")
            efficiency=$(awk "BEGIN {printf \"%.2f\", ($BASELINE_TIME_STRONG / $time_ms) / $cores * 100}")
        fi
    else
        # Weak scaling: tempo ideale dovrebbe rimanere costante
        if [ $BASELINE_TIME_WEAK -eq 0 ]; then
            BASELINE_TIME_WEAK=$time_ms
            speedup=1.00
            efficiency=100.00
        else
            # Per weak scaling, speedup = cores (ideale)
            # Efficiency = tempo_baseline / tempo_corrente * 100
            speedup=$(awk "BEGIN {printf \"%.2f\", $cores * 1.0}")
            efficiency=$(awk "BEGIN {printf \"%.2f\", $BASELINE_TIME_WEAK / $time_ms * 100}")
        fi
    fi

    echo "Speedup: ${speedup}x | Efficiency: ${efficiency}%" | tee -a $output_file
    echo "===========================================" | tee -a $output_file

    # Salva in CSV
    echo "$SIZE_MODE,$scaling_type,$mode,$cores,$mpi,$omp,$nodes,$time_ms,$speedup,$efficiency,$layer_size" >> $CSV_FILE

    return 0
}

# ================= STRONG SCALING TESTS =================
echo ""
echo "========================================"
echo "PARTE 1: STRONG SCALING"
echo "Problema fisso, aumentiamo i cores"
echo "========================================"

# Reset baseline
BASELINE_TIME_STRONG=0

for cores in "${CORES_TO_TEST[@]}"; do
    echo ""
    echo "=============================="
    echo "TESTING $cores CORES - STRONG SCALING"
    echo "=============================="

    # 1. MPI PURO
    echo ""
    echo ">> MPI PURE: $cores ranks x 1 thread"
    nodes=$(( (cores + 31) / 32 ))  # Calcola nodi necessari
    run_test $cores $cores 1 $nodes "MPI-Pure" "Strong" $BASE_LAYER_SIZE $BASE_TARGET_PARTICLES

    # 2. MPI+OMP HYBRID
    echo ""
    read mpi omp nodes <<< $(get_hybrid_config $cores)
    echo ">> MPI+OMP HYBRID: $mpi ranks x $omp threads"
    run_test $cores $mpi $omp $nodes "Hybrid" "Strong" $BASE_LAYER_SIZE $BASE_TARGET_PARTICLES

    # Pausa tra test
    sleep 2
done

# ================= WEAK SCALING TESTS =================
echo ""
echo "========================================"
echo "PARTE 2: WEAK SCALING"
echo "Problema scala con cores, tempo dovrebbe rimanere costante"
echo "========================================"

# Reset baseline per weak scaling
BASELINE_TIME_WEAK=0

for cores in "${CORES_TO_TEST[@]}"; do
    echo ""
    echo "=============================="
    echo "TESTING $cores CORES - WEAK SCALING"
    echo "=============================="

    # Calcola dimensioni problema scalate
    weak_layer_size=$(get_weak_scaling_layer_size $cores)
    weak_particles=$(get_weak_scaling_particles $cores)

    echo "Scaled problem: Layer=$weak_layer_size, Particles=$weak_particles"

    # 1. MPI PURO
    echo ""
    echo ">> MPI PURE: $cores ranks x 1 thread"
    nodes=$(( (cores + 31) / 32 ))
    run_test $cores $cores 1 $nodes "MPI-Pure" "Weak" $weak_layer_size $weak_particles

    # 2. MPI+OMP HYBRID
    echo ""
    read mpi omp nodes <<< $(get_hybrid_config $cores)
    echo ">> MPI+OMP HYBRID: $mpi ranks x $omp threads"
    run_test $cores $mpi $omp $nodes "Hybrid" "Weak" $weak_layer_size $weak_particles

    # Pausa tra test
    sleep 2
done

# ================= SUMMARY =================
echo ""
echo "========================================"
echo "TESTS COMPLETATI!"
echo "========================================"
echo "Results directory: $RESULTS_DIR"
echo "CSV file: $CSV_FILE"
echo ""

# Mostra summary
echo "=== STRONG SCALING SUMMARY ==="
echo ""
grep "Strong" $CSV_FILE | column -t -s,

echo ""
echo "=== WEAK SCALING SUMMARY ==="
echo ""
grep "Weak" $CSV_FILE | column -t -s,

echo ""
echo "Usa il notebook Jupyter per analisi grafiche complete!"
echo "Done!"
