To: carc-support@usc.edu
Subject: Discovery: 19 jobs in the gpu partition show state CG (completing)

Hi CARC team,

I'm reporting what I see in squeue. I don't know the cause.

At Mon Oct 12 08:44 PDT, on a Discovery login node, I ran:

squeue -p gpu -t CG

It printed:

             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          12922613       gpu gesture-   binhan CG      30:19      1 b17-14
          12885864       gpu   gsseed peisengo CG    3:00:12      1 a03-20
          12902223       gpu gen3cSWd peisengo CG   12:00:04      1 b05-11
          12902221       gpu   gsseed peisengo CG    3:00:15      1 a03-01
          12902220       gpu   gsseed peisengo CG    3:00:29      1 a03-01
          12931905       gpu     bash juannkim CG    4:00:06      1 b18-16
          12923957       gpu     bash juannkim CG    4:00:22      1 c04-02
          12923536       gpu     bash juannkim CG      24:41      1 b17-16
          12922716       gpu ard-k1-r muttreja CG    7:31:51      1 d14-07
          12922717       gpu ard-k2-r muttreja CG    7:31:51      1 e21-04
          12922718       gpu ard-k3-r muttreja CG    7:31:51      1 e21-01
          12922725       gpu ard-k10- muttreja CG    7:31:51      1 e21-15
          12922727       gpu ard-k12- muttreja CG    7:31:51      1 e22-01
          12922730       gpu ard-k15- muttreja CG    7:31:51      1 e22-04
          12922731       gpu ard-k16- muttreja CG    7:31:51      1 d23-14
          12939035       gpu tcn5-pai bowenson CG    4:00:07      1 e21-09
          12939034       gpu tcn5-pai bowenson CG    4:00:10      1 e22-12
          12922947       gpu submit_b  pranavj CG   18:03:28      1 d23-14
          12923554       gpu     bash juannkim CG    4:00:16      1 b18-18

I ran the same command a few minutes earlier, at about 08:41, and got the same 19 jobs.

Thanks,
Matthew Gordy (mgordy), account xiangren_1987
