void f() {
loop:
	do_something();

	if (condition) {
		goto loop;
	}
}

void f() {
	while (1) {
		do_something();

		if (condition)
			continue_to loop_continue;

		break_to loop_break;
	loop_continue:
	}
loop_break:
}
