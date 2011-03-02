#ifndef __ERRCODE_H
#define __ERRCODE_H

/* Warning - side effects */

#define ECDIGIT(str, d) (str?str[d]-'0':0)
#define ECFAIL(str)     (ECDIGIT(str, 0)>2)
#define ECAGAIN(str)    (ECDIGIT(str, 0)==4)
#define ECWARN(str)     (ECDIGIT(str, 0)>1)
#define ECFATAL(str)    (ECDIGIT(str, 0)>4)

#define ECRETRY(var, expr, max) for(int __retry=max; (var=expr) && ECAGAIN(var) && __retry; __retry--)

#ifdef __cplusplus

inline int ecdigit(const char *str, int d) { return str ? str[d]-'0' : 0; }
inline int ecfail(const char *str) { return ecdigit(str, 0) > 2; }
inline int ecagain(const char *str) { return ecdigit(str, 0) == 4; }
inline int ecwarn(const char *str) { return ecdigit(str, 0) > 1; }
inline int ecfatal(const char *str) { return ecdigit(str, 0) > 4; }

inline int ecparameter(const char *str) {return ecdigit(str, 1) == 0;}
inline int eccommunications(const char *str) {return ecdigit(str, 1) == 1;}
inline int ecfilesystem(const char *str) {return ecdigit(str, 1) == 2;}
inline int ecip(const char *str) {return ecdigit(str, 1) == 3;}
inline int ecsensor(const char *str) {return ecdigit(str, 1) == 4;}
inline int ecmotion(const char *str) {return ecdigit(str, 1) == 5;}
inline int ecconveyor(const char *str) {return ecdigit(str, 1) == 6;}
inline int ecos(const char *str) {return ecdigit(str, 1) == 7;}
inline int ecreserved(const char *str) {return ecdigit(str, 1) == 8;}
inline int ecother(const char *str) {return ecdigit(str, 1) == 9;}

#endif

#endif

/*

Proposed Error Code Schema
1/28/2000

1 Format
 
 "xyz text message"

  where x, y, and z are ASCII digits and text message is an
  end-user-readable ASCIIZ string with no ending linefeed.

  The single digit x represents the error "disposition" and can be
  used to determine how to deal with an error. The second digit
  indicates an error "classification" and can be used to identify
  error sources. The third digit may be used to uniquely identify
  errors for reporting purposes, though the text message should
  suffice. Use '0' in the third digit to indicate that no effort was
  made to disambiguate the numeric error code.

2 Disposition

 0yz Success

  Function was successful. Simply return NULL instead. 

 1yz Notice

  Nonfatal problems. System performed action anyway.  Message may or
  may not be rippled up and displayed as status without
  interruption. Implementation optional.

 2yz Warning

  Nonfatal problems. System performed action anyway.  Message should
  be shown to user, logged, or ignored at caller's disgression (and
  peril), but the system should proceed without interruption.

  Fiducial alignment out of tolerance
  Dark images
  Lower level retry succeeded
  Image queue was nearly overrun
  CRC errors on non-critical data (fiducial template?)

 3yz Reserved nonfatal

  Treated as failure. Don't use.

 4yz Transient error

  Command may be retried without change. 
  Caller should do one of the following:
   Retry
   Retry and log
   Retry and return notice
   Stop and return error intact

  Communication timeout
  Disk or queue full
  Swath missed a strobe
  Timeout waiting on sensor data

 5yz User or service intervention

  Something that should have worked failed for reasons unknown. System
  must restart to return to known state. A caller may accomplish this
  by rippling the error code up or if necessary jumping to a restart
  hook (via signal, setjump, etc.)

  Hardware communication failed
  Motion stall
  System in passive mode
  Retry limit exceeded
  EMO

 6yz Permanent error 

  The EBADUSER error - you tried something that can't work. Bad
  parameters, syntax, etc. Generally programmer error. These shouldn't
  happen and when they do, the system can be assumed to be in an
  unknown state as the programmer has made a mistake. The lights are
  out. If you continue, you will likely be eaten by a grue. Therefore,
  it's best to restart. If restarting's not appropriate, make it a
  warning.

  Axis already in motion
  Syntax error in schedule
  Something segfaulted
  Something will segfault if I do what you asked
  Image referenced in schedule not found
  CRC error on critical data (calibration image?)
  Feature unimplemented
  Command out of sequence (move before home?)
  Requested data not available

 7yz Reserved fatal

  End of the world. Don't use.

3 Classification

 Informational error source. May be used to classify errors for
 logging, and perhaps to ignore a set of errors for testing purposes,
 but otherwise not used for policy decisions. Errors should be
 preferentially be put into earlier categories, ie motion syntax
 errors should be classified as syntax errors.

 x0z Syntax or parameter
 x1z Communication
 x2z Filesystem
 x3z Image processing
 x4z Sensor
 x5z Motion
 x6z Conveyor
 x7z Operating system
 x8z Reserved
 x9z Other

4 Using Error Codes

 With the help of a few macros, errors can be easily dealt with
 appropriately:

  ECFAIL(cond)     true if function was unsuccessful (x>2)
  ECAGAIN(cond)    true if function had transient failure (x==4)
  ECWARN(cond)	   true if we bother noting error (x>1)
  ECFATAL(cond)	   true if we should restart (x>4)	  
 
 Example:

  const char *foo()
  {
	  err=0;

	  for(int retry=max;
	      ECAGAIN(err=dosomething()) && retry;
	      retry--)
		  log("Retrying..", err); // logging optional

	  if(ECFAIL(err)) return err; // either out of retries or fatal error 

	  if(bar) err=dosomethingelse(); // discard earlier nonfatal errors

	  return err;
  }

 The retry loop can be wrapped in a macro like this:

  ECRETRY(err=dosomething(), max);

 Note that error strings should probably not be automatics and not
 dynamically created. No one will free them for you. 

----------

What follows is from RFC 822 - Simple Mail Transfer Protocol

   4.2.1.  REPLY CODES BY FUNCTION GROUPS

      500 Syntax error, command unrecognized
         [This may include errors such as command line too long]
      501 Syntax error in parameters or arguments
      502 Command not implemented
      503 Bad sequence of commands
      504 Command parameter not implemented

      211 System status, or system help reply
      214 Help message
         [Information on how to use the receiver or the meaning of a
         particular non-standard command; this reply is useful only
         to the human user]

      220 <domain> Service ready
      221 <domain> Service closing transmission channel
      421 <domain> Service not available,
          closing transmission channel
         [This may be a reply to any command if the service knows it
         must shut down]

      250 Requested mail action okay, completed
      251 User not local; will forward to <forward-path>
      450 Requested mail action not taken: mailbox unavailable
         [E.g., mailbox busy]
      550 Requested action not taken: mailbox unavailable
         [E.g., mailbox not found, no access]
      451 Requested action aborted: error in processing
      551 User not local; please try <forward-path>
      452 Requested action not taken: insufficient system storage
      552 Requested mail action aborted: exceeded storage allocation
      553 Requested action not taken: mailbox name not allowed
         [E.g., mailbox syntax incorrect]
      354 Start mail input; end with <CRLF>.<CRLF>
      554 Transaction failed
...

APPENDIX E

Theory of Reply Codes

   The three digits of the reply each have a special significance.
   The first digit denotes whether the response is good, bad or
   incomplete.  An unsophisticated sender-SMTP will be able to
   determine its next action (proceed as planned, redo, retrench,
   etc.) by simply examining this first digit.  A sender-SMTP that
   wants to know approximately what kind of error occurred (e.g., mail
   system error, command syntax error) may examine the second digit,
   reserving the third digit for the finest gradation of information.

      There are five values for the first digit of the reply code:

         1yz   Positive Preliminary reply

            The command has been accepted, but the requested action is
            being held in abeyance, pending confirmation of the
            information in this reply.  The sender-SMTP should send
            another command specifying whether to continue or abort
            the action.

               [Note: SMTP does not have any commands that allow this
               type of reply, and so does not have the continue or
               abort commands.]

         2yz   Positive Completion reply

            The requested action has been successfully completed.  A
            new request may be initiated.

         3yz   Positive Intermediate reply

            The command has been accepted, but the requested action is
            being held in abeyance, pending receipt of further
            information.  The sender-SMTP should send another command
            specifying this information.  This reply is used in
            command sequence groups.

         4yz   Transient Negative Completion reply

            The command was not accepted and the requested action did
            not occur.  However, the error condition is temporary and
            the action may be requested again.  The sender should
            return to the beginning of the command sequence (if any).
            It is difficult to assign a meaning to "transient" when
            two different sites (receiver- and sender- SMTPs) must
            agree on the interpretation.  Each reply in this category
            might have a different time value, but the sender-SMTP is
            encouraged to try again.  A rule of thumb to determine if
            a reply fits into the 4yz or the 5yz category (see below)
            is that replies are 4yz if they can be repeated without
            any change in command form or in properties of the sender
            or receiver.  (E.g., the command is repeated identically
            and the receiver does not put up a new implementation.)

         5yz   Permanent Negative Completion reply

            The command was not accepted and the requested action did
            not occur.  The sender-SMTP is discouraged from repeating
            the exact request (in the same sequence).  Even some
            "permanent" error conditions can be corrected, so the
            human user may want to direct the sender-SMTP to
            reinitiate the command sequence by direct action at some
            point in the future (e.g., after the spelling has been
            changed, or the user has altered the account status).

      The second digit encodes responses in specific categories:

         x0z   Syntax -- These replies refer to syntax errors,
               syntactically correct commands that don't fit any
               functional category, and unimplemented or superfluous
               commands.

         x1z   Information --  These are replies to requests for
               information, such as status or help.

         x2z   Connections -- These are replies referring to the
               transmission channel.

         x3z   Unspecified as yet.

         x4z   Unspecified as yet.

         x5z   Mail system -- These replies indicate the status of
               the receiver mail system vis-a-vis the requested
               transfer or other mail system action.

      The third digit gives a finer gradation of meaning in each
      category specified by the second digit.  The list of replies
      illustrates this.  Each reply text is recommended rather than
      mandatory, and may even change according to the command with
      which it is associated.  On the other hand, the reply codes must
      strictly follow the specifications in this section.  Receiver
      implementations should not invent new codes for slightly
      different situations from the ones described here, but rather
      adapt codes already defined.

      For example, a command such as NOOP whose successful execution
      does not offer the sender-SMTP any new information will return a
      250 reply.  The response is 502 when the command requests an
      unimplemented non-site-specific action.  A refinement of that is
      the 504 reply for a command that is implemented, but that
      requests an unimplemented parameter.

*/
