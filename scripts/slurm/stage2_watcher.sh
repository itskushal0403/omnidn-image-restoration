#!/bin/bash
# Login-node watcher: submits each stage-2 pipeline as soon as the stage-1 run(s) it needs have finished
# (needed because the mtech QoS caps submitted jobs at 3).  Run with nohup; log in logs/stage2_watcher.log
cd /csehome/m25csa016/denoising_competition
M_DONE=0; S_DONE=0
while [ $M_DONE = 0 ] || [ $S_DONE = 0 ]; do
  if [ $M_DONE = 0 ] && [ -f checkpoints/omnidn_m_stage1/ema_final.pth ]; then
    sbatch --job-name=dn-ftM scripts/slurm/finetune_pipeline.sbatch configs/omnidn_m_ft_fid.yaml configs/omnidn_m_ft_perc.yaml m && M_DONE=1
    echo "$(date) submitted M pipeline"
  fi
  # (S distillation from M was dropped after ablation C: same wall-clock, lower PSNR)
  if [ $S_DONE = 0 ] && [ -f checkpoints/omnidn_s_stage1/ema_final.pth ]; then
    sbatch --job-name=dn-ftS scripts/slurm/finetune_pipeline.sbatch configs/omnidn_s_ft_fid.yaml configs/omnidn_s_ft_perc.yaml s && S_DONE=1
    echo "$(date) submitted S pipeline"
  fi
  sleep 60
done
echo "$(date) watcher done"
