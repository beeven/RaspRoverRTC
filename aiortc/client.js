var pc = null;

async function negotiate() {
    const micStream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: false
    });
    for (const track of micStream.getAudioTracks()) {
        console.log(track);
        pc.addTrack(track);
    }

    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'sendrecv' });


    let offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    await (new Promise(resolve => {
        if (pc.iceGatheringState === 'complete') { resolve(); }
        else { pc.addEventListener('icegatheringstatechange', () => pc.iceGatheringState === 'complete' && resolve()) }
    }));

    document.getElementById("offer").textContent = pc.localDescription.sdp;

    let resp = await fetch('/offer', {
        body: JSON.stringify({
            sdp: offer.sdp,
            type: offer.type,
        }),
        headers: {
            'Content-Type': 'application/json'
        },
        method: 'POST'
    });

    let answer = await resp.json();
    document.getElementById("answer").textContent = answer.sdp;
    await pc.setRemoteDescription(answer);




    // return pc.createOffer().then((offer) => {
    //     return pc.setLocalDescription(offer);
    // }).then(() => {
    //     // wait for ICE gathering to complete
    //     return new Promise((resolve) => {
    //         if (pc.iceGatheringState === 'complete') {
    //             resolve();
    //         } else {
    //             const checkState = () => {
    //                 if (pc.iceGatheringState === 'complete') {
    //                     pc.removeEventListener('icegatheringstatechange', checkState);
    //                     resolve();
    //                 }
    //             };
    //             pc.addEventListener('icegatheringstatechange', checkState);
    //         }
    //     });
    // }).then(() => {
    //     var offer = pc.localDescription;
    //     document.getElementById("offer").textContent = offer.sdp;
    //     return fetch('/offer', {
    //         body: JSON.stringify({
    //             sdp: offer.sdp,
    //             type: offer.type,
    //         }),
    //         headers: {
    //             'Content-Type': 'application/json'
    //         },
    //         method: 'POST'
    //     });
    // }).then((response) => {
    //     return response.json();
    // }).then((answer) => {
    //     document.getElementById("answer").textContent = answer.sdp;
    //     return pc.setRemoteDescription(answer);
    // }).catch((e) => {
    //     alert(e);
    // });
}

async function start() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    if (document.getElementById('use-stun').checked) {
        config.iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }];
    }

    pc = new RTCPeerConnection(config);

    // connect audio / video
    pc.addEventListener('track', (evt) => {
        if (evt.track.kind == 'video') {
            document.getElementById('video').srcObject = evt.streams[0];
        } else {
            document.getElementById('audio').srcObject = evt.streams[0];
        }
    });

    document.getElementById('start').style.display = 'none';
    await negotiate();
    document.getElementById('stop').style.display = 'inline-block';
}

async function stop() {
    document.getElementById('stop').style.display = 'none';

    // close peer connection
    setTimeout(() => {
        pc.close();
    }, 500);
}
