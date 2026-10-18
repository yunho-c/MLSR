#include "mlsr.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

static MlsrString str(const char *s) { MlsrString value={(const uint8_t *)s,strlen(s)};return value; }
static void ok(MlsrStatus status, MlsrHandle error) {
    if(status==MLSR_OK) return;
    MlsrErrorInfo view={0};
    if(mlsr_error_read(error,&view,sizeof(view))==MLSR_OK)
        fprintf(stderr,"%.*s: %.*s\n",(int)view.code.len,view.code.data,(int)view.message.len,view.message.data);
    mlsr_error_release(error);
    assert(status==MLSR_OK);
}
#ifdef _WIN32
static DWORD WINAPI cancel_thread(void *pointer) {
#else
static void *cancel_thread(void *pointer) {
#endif
    MlsrHandle error=0;
    MlsrStatus status=mlsr_job_cancel(*(MlsrHandle *)pointer,&error);
    ok(status,error);return 0;
}
int main(int argc,char **argv) {
    assert(argc==3 || argc==4);
    const uint64_t initial=mlsr_live_handles();
    MlsrHandle engine=0,job=0,error=0,event=0,retained=0;
    MlsrEngineOptions engine_options;
    assert(mlsr_engine_options_default(&engine_options)==MLSR_OK);
    assert(mlsr_features() & MLSR_FEATURE_JOBS);
    assert(mlsr_features() & MLSR_FEATURE_MODEL_STORE);
    if(argc==4){assert(strcmp(argv[3],"qnn_htp")==0);engine_options.compute_policy=2;assert(mlsr_features()&MLSR_FEATURE_QNN_HTP);}
    printf("features=%u\n", mlsr_features());
    MlsrStatus status=mlsr_engine_open(str(argv[1]),str(argv[2]),&engine_options,&engine,&error);ok(status,error);
    MlsrJobOptions options;mlsr_job_options_default(&options);
    options.queue_frames=4096;options.chunk_frames=1024;
    status=mlsr_job_start(engine,str("Hello from MLSR.\nThis is a second sentence."),str("en"),str("M1"),&options,&job,&error);ok(status,error);
    uint64_t frames=0;unsigned predictions=0,terminals=0;
    for(;;) {
        status=mlsr_job_poll(job,30000,&event,&error);
        if(status==MLSR_END)break;
        ok(status,error);
        MlsrEventInfo info={0};status=mlsr_event_read(event,&info,sizeof(info),&error);ok(status,error);
        assert(info.job_id==mlsr_job_id(job));
        if(info.kind==1){predictions++;assert(info.predicted_frames>0);}
        else if(info.kind==2){
            assert(info.sample_rate==44100 && info.channels==1 && info.start_frame==frames);
            assert(info.sample_count==info.frames && info.frames<=1024);
            for(uintptr_t i=0;i<info.sample_count;i++) assert(isfinite(info.samples[i]));
            frames+=info.frames;
            if(!retained){retained=event;event=0;}
        } else {assert(info.kind==3 && info.frames==frames && info.completed_segments==2);terminals++;}
        assert(mlsr_event_release(event)==MLSR_OK);
    }
    assert(frames>44100 && predictions==2 && terminals==1);
    assert(mlsr_job_release(job)==MLSR_OK);
    // A one-frame queue cannot complete while the consumer is stopped.
    options.queue_frames=1;options.chunk_frames=1;
    status=mlsr_job_start(engine,str("안녕하세요. 다시 읽습니다."),str("ko"),str("F1"),&options,&job,&error);ok(status,error);
    MlsrHandle busy=123;
    status=mlsr_job_start(engine,str("Busy."),str("en"),str("M1"),&options,&busy,&error);
    assert(status==MLSR_ERROR && busy==0);assert(mlsr_error_release(error)==MLSR_OK);
    status=mlsr_job_poll(job,30000,&event,&error);ok(status,error);mlsr_event_release(event);
#ifdef _WIN32
    HANDLE thread=CreateThread(NULL,0,cancel_thread,&job,0,NULL);assert(thread);
    assert(WaitForSingleObject(thread,30000)==WAIT_OBJECT_0);CloseHandle(thread);
#else
    pthread_t thread;assert(pthread_create(&thread,NULL,cancel_thread,&job)==0);assert(pthread_join(thread,NULL)==0);
#endif
    status=mlsr_job_wait(job,30000,&error);ok(status,error);
    status=mlsr_job_poll(job,0,&event,&error);ok(status,error);
    MlsrEventInfo info={0};status=mlsr_event_read(event,&info,sizeof(info),&error);ok(status,error);assert(info.kind==4);
    mlsr_event_release(event);assert(mlsr_job_release(job)==MLSR_OK);
    assert(mlsr_engine_release(engine)==MLSR_OK);
    // PCM retained as an event survives destruction of job and engine.
    status=mlsr_event_read(retained,&info,sizeof(info),&error);ok(status,error);
    assert(info.kind==2 && info.sample_count>0 && isfinite(info.samples[0]));
    assert(mlsr_event_release(retained)==MLSR_OK);
    assert(mlsr_live_handles()==initial);
    printf("Synthesis C ABI passed: %llu frames, cancellation and owned lifetime\n",(unsigned long long)frames);
    return 0;
}
