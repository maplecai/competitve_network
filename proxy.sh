# 23456 - the port not be used
ssh -CqfND 23456 hxcai@166.111.130.51
# setting a proxy service in socks5 protocol

# setting terminal proxy

#export http_proxy=socks5://127.0.0.1:23456
#export https_proxy=socks5://127.0.0.1:23456
#export all_proxy=socks5://127.0.0.1:23456

export all_proxy=socks5://127.0.0.1:23456

# using proxy in pip
# pip install numpy==1.21.5
curl www.baidu.com

lsof -i :23456
