/*
 * StopWatch.h
 *
 *  Created on: Nov 18, 2019
 *      Author: himesb
 */


namespace cistem_timer_noop {

class StopWatch {


public:

	StopWatch();
	virtual ~StopWatch();

	// dummy methods
	inline void start(std::string name) {return;}
	inline void lap(std::string name) {return;}
	inline void check_for_name_and_set_current_idx(std::string name) {return;}
	inline void mark_entry_or_exit_point() {return;}

};

} // namespace cistem_timer_noop

namespace cistem_timer {
class StopWatch {


public:

	StopWatch();
	virtual ~StopWatch();


	// Create or reuse event named "name", start timing. May overlap with other timing events.
	void start(std::string name);

	// Record the elapsed time since last "start" for this event. Add to cummulative time.
	void lap(std::string name);

	// Print out all event times, including stopwatch overhead time.
	void print_times();

	// Start or pause the total elapsed time when passing a stopwatch pointer to a method. Place inside the method at the entry and exit point of the method call.
	void mark_entry_or_exit_point();






private:

	enum TimeFormat : int { NANOSECONDS, MICROSECONDS, MILLISECONDS, SECONDS };
	typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_pt;


	std::vector<std::string> event_names = {};
	std::vector<time_pt> 	 event_times = {};
	std::vector<uint64_t> 	 elapsed_times = {};

	uint64_t hrminsec[4] = {0,0,0,0};
	size_t current_index;
	uint64_t null_time;
	bool is_new;
	bool is_set_overall;
	TimeFormat time_fmt = MICROSECONDS;

	uint64_t stop(TimeFormat T, int idx);
	uint64_t ticks(TimeFormat T, const time_pt& start_time, const time_pt& end_time);

	// Parse time into a more readable hours:minutes:seconds:milliseconds format for display.
	void convert_time(uint64_t microsec);

		// Check to see if an event named "name" already exists, if not, initialize it.
	void check_for_name_and_set_current_idx(std::string name);

};
} // namespace cistem_timer