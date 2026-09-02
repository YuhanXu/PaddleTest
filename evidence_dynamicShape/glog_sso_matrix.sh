set +u
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PYTHONPATH=/work/Paddle/build/python
source /work/env3.10/bin/activate
export GLOG_logbufsecs=0
for VM in "a=1" "abcdefghijklm=1" "abcdefghijklmn=1" "abcdefghijklmno=1" "abcdefghijklmnop=1" "shape_optimization_pass=3"; do
  GLOG_vmodule="$VM" timeout -s KILL 60 python -c "import paddle" > /tmp/sso.log 2>&1
  rc=$?
  echo "len=${#VM} vmodule='$VM' EXIT=$rc  err=$(grep -m1 -E 'corrupt|double free|malloc_consolidate|free\(\)' /tmp/sso.log | head -c 60)"
done
