echo "Running your GA Attacker"

'''
for i in {2..10}; do
    ((i = i*100))
    for k in {1..10}; do 
        ((k = k/20000.0)); echo $k;
        echo "Population = $i -------------- Mutation = $k"
        #python simpleExploratoryAttacker.py $i $k
    done
done
'''


for i in {1..39}; do
    ((k = $i*100))
    python simpleExploratoryAttacker.py $k 0.005
done

echo " I am done running your GA attacker."
