# import logging
# import sys
#
# import aiy.assistant.auth_helpers
# import aiy.voicehat
# from google.assistant.library import Assistant
# from google.assistant.library.event import EventType
#
# import aiy.cloudspeech
# import aiy.audio
import time
import requests

# logging.basicConfig(
#     level=logging.INFO,
#     format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
# )


# def process_event(event):
#     status_ui = aiy.voicehat.get_status_ui()
#     if event.type == EventType.ON_START_FINISHED:
#         status_ui.status('ready')
#         if sys.stdout.isatty():
#             print('Say "OK, Google" then speak, or press Ctrl+C to quit...')
#
#     elif event.type == EventType.ON_CONVERSATION_TURN_STARTED:
#         status_ui.status('listening')
#
#
#     elif event.type == EventType.ON_END_OF_UTTERANCE:
#         status_ui.status('thinking')
#
#     elif event.type == EventType.ON_CONVERSATION_TURN_FINISHED:
#         status_ui.status('ready')
#
#     elif event.type == EventType.ON_ASSISTANT_ERROR and event.args and event.args['is_fatal']:
#         sys.exit(1)


def getCryptoStatus():
    neoEurPrice = float(requests.get('https://api.cryptonator.com/api/ticker/neo-eur').json()['ticker']['price'])
    time.sleep(0.100)
    ltcEurPrice = float(requests.get('https://api.cryptonator.com/api/ticker/ltc-eur').json()['ticker']['price'])
    time.sleep(0.100)
    iotEurPrice = float(requests.get('https://api.cryptonator.com/api/ticker/iot-eur').json()['ticker']['price'])
    time.sleep(0.100)
    ethEurPrice = float(requests.get('https://api.cryptonator.com/api/ticker/eth-eur').json()['ticker']['price'])
    time.sleep(0.100)
    grcEurPrice = float(requests.get('https://api.cryptonator.com/api/ticker/grc-eur').json()['ticker']['price'])
    time.sleep(0.100)
    btcEurPrice = float(requests.get('https://api.cryptonator.com/api/ticker/btc-eur').json()['ticker']['price'])
    time.sleep(0.100)
    scEurPrice = float(requests.get('https://api.cryptonator.com/api/ticker/sc-eur').json()['ticker']['price'])

    neoHoldings = 3.8302
    ltcHoldings = 1.6456
    iotHoldings = 191.2439
    ethHoldings = 0.38350873
    grcHoldings = 1093.35053161
    btcHoldings = 0.02763516
    scHoldings = 4030.86

    oldNeoEurPrice = 20.73
    oldLtcEurPrice = 58.27
    oldIotEurPrice = 0.4887
    oldEthEurPrice = 250.32
    oldGrcEurPrice = 0.0936900096994702
    oldBtcEurPrice = 9562.0577548
    oldScEurPrice = 0.024831

    oldNeoEurBalance = neoHoldings * oldNeoEurPrice
    oldLtcEurBalance = ltcHoldings * oldLtcEurPrice
    oldIotEurBalance = iotHoldings * oldIotEurPrice
    oldEthEurBalance = ethHoldings * oldEthEurPrice
    oldGrcEurBalance = grcHoldings * oldGrcEurPrice
    oldBtcEurBalance = btcHoldings * oldBtcEurPrice
    oldScEurBalance = scHoldings * oldScEurPrice

    currentNeoEurBalance = neoHoldings * neoEurPrice
    currentLtcEurBalance = ltcHoldings * ltcEurPrice
    currentIotEurBalance = iotHoldings * iotEurPrice
    currentEthEurBalance = ethHoldings * ethEurPrice
    currentGrcEurBalance = grcHoldings * grcEurPrice
    currentBtcEurBalance = btcHoldings * btcEurPrice
    currentScEurBalance = scHoldings * scEurPrice

    # oldEurTotal = oldNeoEurBalance + oldLtcEurBalance + oldIotEurBalance + oldEthEurBalance + oldGrcEurBalance + oldBtcEurBalance # Need?
    oldEurTotal = 700
    currentEurTotal = currentNeoEurBalance + currentLtcEurBalance + currentIotEurBalance + currentEthEurBalance + currentGrcEurBalance + currentBtcEurBalance
    totalDifference = currentEurTotal - oldEurTotal

    returningText = []
    returningText.append("Current balance is: " + str(int(currentEurTotal)))
    if totalDifference > 0:
        returningText.append("Winning: " + str(int(totalDifference)) + " euros, ")
        returningText.append(str(int((currentEurTotal / float(oldEurTotal) - 1) * 100)) + " %.")
    else:
        returningText.append("Losing: " + str(int(totalDifference * -1)) + " euros, ")
        returningText.append(str(int((oldEurTotal / float(currentEurTotal) - 1) * 100)) + " %.")

    return returningText


# def main():
#     credentials = aiy.assistant.auth_helpers.get_assistant_credentials()
#
#     with Assistant(credentials) as assistant:
#         for event in assistant.start():
#             if event.type == EventType.ON_RECOGNIZING_SPEECH_FINISHED:
#
#                 if event.args['text'] == 'crypto':
#                     cryptoStatus = getCryptoStatus()
#                     time.sleep(1)  # Let regular google assistant give the erroneous response first
#                     aiy.audio.say(cryptoStatus[0])
#                     aiy.audio.say(cryptoStatus[1])
#                     aiy.audio.say(cryptoStatus[2])
#
#             else:
#                 process_event(event)


if __name__ == '__main__':
    # main()
    cryptoStatus = getCryptoStatus()
    print("cryptoStatus: {}".format(cryptoStatus))