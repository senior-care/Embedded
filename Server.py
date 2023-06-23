import socket 
import cv2
import numpy
import datetime
import base64
import time
import threading
import collections
import queue
import pymysql  
import boto3
import re
from datetime import datetime
import logging as log
import uuid


class ServerSocket:

    def __init__(self, socket, lock):
        super().__init__()
        
        self.sock = socket
        self.lock = lock

    def run(self):
        self.receiveThread = threading.Thread(target=self.receiveImages)
        self.receiveThread.daemon = True
        self.receiveThread.start()


    def socketClose(self):
        self.sock.close()


    def receiveImages(self):
        global store_queue #공유변수

        log.info("receive 스레드 시작")
        try:
            while True:
                
                log.info('데이터 기다리는 중')

                length = self.recvall(self.sock, 64)
                length1 = length.decode('utf-8')
                stringData = self.recvall(self.sock, int(length1)) 
                log.info("노인 이미지 받기 완료")
                stime = self.recvall(self.sock, 64)
                log.info('노인 정보 받기 완료')

                #now = datetime.now()
                #recievetime = now.strftime('%Y-%m-%d %H:%M:%S.%f')
                #print('송신 시각 : ' + str(recievetime))
                person_info = stime.decode('utf-8')
                data = numpy.frombuffer(base64.b64decode(stringData), numpy.uint8)
                decimg = cv2.imdecode(data, 1) 

                #공유변수에 값 저장
                #print('데이터 저장중...')
                self.lock.acquire()
                store_queue.put([person_info, decimg])
                self.lock.release()
                log.info('공유 변수에 데이터 저장 완료')

                # print("===========================")
                # print(store_queue.queue)
                time.sleep(0.01)

        except Exception as e:
            print(e)
            self.socketClose()
            #cv2.destroyAllWindows()
            delete_socket(self) #소켓리스트에서 제거

    def recvall(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf


#클라이언트 소켓 관리
def create_socket(socket, lock):
    global socket_list

    S = ServerSocket(socket, lock)
    socket_list.append(S)
    index = socket_list.index(S)
    socket_list[index].deamon = True
    socket_list[index].run()

def delete_socket(socketClass):
    global socket_list
    socket_list.remove(socketClass)


#DB 전송 스레드
def send(lock, conn, s3):
    log.info('DB 전송 스레드 시작')
    global store_queue
    while True:
        try:
            if(store_queue.qsize()>0):

                log.info('DB 전송 시작')
                lock.acquire()
                temp, decimg =store_queue.get()
                print(temp)
                #server 에서 데이터 파싱
                detect_date, detect_time, seniorno, emotion_code = temp.split()
                
                #server 에서 s3 이미지 처리
                #로컬에 저장
                t_file_name = str(uuid.uuid1())
                log.info(t_file_name)
                path='E:/rnqhstlr/temp/Image/'+ t_file_name + '.jpg' 
                cv2.imwrite(path, decimg)
                log.info('로컬 저장소에 이미지 저장 완료')

                #S3에 저장
                bucket = 'imagestorge'  #s3의 버킷명
                accesss_key = t_file_name + '.jpg'  #s3에 저장되는 파일명
                s3.upload_file(path, bucket, accesss_key)
                log.info('S3 저장소에 전송 완료')
                #S3 URL 알아내기
                image_path = 'https://'+bucket+'.s3.ap-northeast-2.amazonaws.com/'+accesss_key

                # 최종 DB 삽입 쿼리
                cur = conn.cursor()
                str1 = "INSERT INTO image (senior_no, image_path, original_name, store_name, detect_date, detect_time, emotion_code) VALUES('" + seniorno +"','" + image_path +"','" + t_file_name +"','" + t_file_name +"','" + detect_date + "','" + detect_time + "','" + emotion_code + "')"
                cur.execute(str1)
                conn.commit()
                log.info('DB 전송 완료')
                lock.release()

            time.sleep(0.1)

        except Exception as e:
            print(e)
            conn.close
    
def s3_connection():
    try:
        s3 = boto3.client (
            service_name = "",
            region_name ="",
            aws_access_key_id = '',
            aws_secret_access_key = ''
        )
    except Exception as e:
        print(e)
    else:
        log.info('S3 연결 success')
        return s3

if __name__ == "__main__":
    
    log.basicConfig(level = 'INFO')
    
    #공유 변수
    socket_list = []
    LOCK=threading.Lock()
    store_queue = queue.Queue()

    #RDS 연결 세팅
    ht = ""
    database = ""
    port = 
    username = ""
    password = ""
    conn = pymysql.connect(
        host=ht, user=username, passwd=password, db=database,
        port=port, use_unicode=True, charset='utf8')  
    log.info('RDS DB 연결 success')

    #S3 연결 세팅
    s3 = s3_connection()

    #AWS(RDS,S3) 전송 스레드 시작
    send_trd = threading.Thread(target=send, args=(LOCK,conn, s3)).start()

    #server 세팅
    ip = ''
    port = 
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM) # 소켓 객체를 생성
    s.bind((ip, port))
    s.listen(10) # 연결 수신 대기 상태(리스닝 수(동시 접속) 설정)


    while True:
        log.info('클라이언트 연결 대기')
        conn, addr = s.accept()

        log.info("클라이언트 연결 완료 addr = %s", addr)
        create_socket(conn, LOCK) #접속한 클라이언트 생성


